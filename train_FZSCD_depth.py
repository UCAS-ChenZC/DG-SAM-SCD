from datasets.change_detection_depth import ChangeDetection_FZSCD
# from models.DT_SCD.BTSCD_1208 import BTSCD as Net
from models.DT_SCD.BTSCD_1208_sam import BTSCD as Net
from utils.palette import color_map_Landsat_SCD as color_map
from utils.metric import IOUandSek
from utils.loss import ChangeSimilarity, DiceLoss, Logit_Interaction_Loss, Logit_Interaction_Loss2
import os
import numpy as np
import torch
from torch.nn import CrossEntropyLoss, BCELoss
from torch.optim import Adam, AdamW, SGD, lr_scheduler
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from models.DT_SCD.hi_lora import LoRA_sam
working_path = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(42)

class Options:
    def __init__(self):
        parser = argparse.ArgumentParser('Semantic Change Detection')
        parser.add_argument("--data_name", type=str, default=r"FZ_SCD_2026")
        parser.add_argument("--Net_name", type=str, default="DT-SCD_Newkd_depDiff_dep")
        parser.add_argument("--backbone", type=str, default="resnet34")
        parser.add_argument("--data_root", type=str, default="/home/solid/CD/datasets/SCD_data/FZ_SCD_2026")
        parser.add_argument("--log_dir", type=str)
        parser.add_argument("--batch_size", type=int, default=16)
        parser.add_argument("--val_batch_size", type=int, default=16)
        parser.add_argument("--test_batch_size", type=int, default=16)
        parser.add_argument("--epochs", type=int, default=100)
        # lr=0.001时，基础网络效果最好，继续增大、lr
        ### czc:保持0.001   观望0.005 ###
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--weight_decay", type=float, default=1e-4)
        parser.add_argument("--lightweight", dest="lightweight", action="store_true",
                           help='lightweight head for fewer parameters and faster speed')
        parser.add_argument("--pretrain_from", type=str,
                           help='train from a checkpoint')
        parser.add_argument("--load_from", type=str,
                           help='load trained model to generate predictions of validation set')
        parser.add_argument("--pretrained", type=bool, default=True,
                           help='initialize the backbone with pretrained parameters')
        parser.add_argument("--tta", dest="tta", action="store_true",
                           help='test_time augmentation')
        parser.add_argument("--warmup", dest="warmup", action="store_true", help='warm up')
        parser.add_argument("--save_mask", dest="save_mask", action="store_true",
                           help='save predictions of validation set during training')
        parser.add_argument("--use_pseudo_label", dest="use_pseudo_label", action="store_true",
                           help='use pseudo labels for re-training (must pseudo label first)')
        parser.add_argument("--M", type=int, default=7)
        ### czc:保持0.0005   观望0.00005 ###
        parser.add_argument("--Lambda", type=float, default=0.0005)
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        print(args)
        return args

class Trainer:
    def __init__(self, args):
    ###     VSCode tensor print setting     ###
        def custom_repr(self):
            return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

        original_repr = torch.Tensor.__repr__
        torch.Tensor.__repr__ = custom_repr
    ###     VSCode tensor print setting     ###
    
        args.log_dir = os.path.join(working_path, 'logs', args.data_name, args.Net_name, args.backbone)
        if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
        self.writer = SummaryWriter(args.log_dir)
        self.args = args

        trainset = ChangeDetection_FZSCD(root=args.data_root, mode="train")
        valset = ChangeDetection_FZSCD(root=args.data_root, mode="val")
        testset = ChangeDetection_FZSCD(root=args.data_root, mode="test")
        self.trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                      pin_memory=False, num_workers=8, drop_last=True)
        self.valloader = DataLoader(valset, batch_size=args.val_batch_size, shuffle=False,
                                    pin_memory=True, num_workers=8, drop_last=False)
        hiera_path = 'pretrained/sam2_hiera_large.pt'
        model = Net(3, num_classes=args.M, checkpoint_path=hiera_path)
        sam_lora = LoRA_sam(model, 16) # LoRA rank设置为16
        self.model = sam_lora.sam
        # self.model = Net(3, num_classes=args.M)
        # self.model = Net(args.backbone, args.pretrained, len(trainset.CLASSES), args.lightweight, args.M, args.Lambda, 80, [2,2,9,2])
        if args.pretrain_from:
            self.model.load_state_dict(torch.load(args.pretrain_from), strict=False)
        if args.load_from:
            self.model.load_state_dict(torch.load(args.load_from), strict=True)

        self.criterion_seg = CrossEntropyLoss(ignore_index=-1)
        self.criterion_bn = CrossEntropyLoss(reduction='none')
        # self.criterion_bn = BCELoss(reduction='none')
        self.criterion_bn_2 = DiceLoss()
        self.criterion_sc = ChangeSimilarity()
        self.kd = Logit_Interaction_Loss2()

        self.optimizer = AdamW([{"params": [param for name, param in self.model.named_parameters()
                                           if "backbone" in name], "lr": args.lr},
                               {"params": [param for name, param in self.model.named_parameters()
                                           if "backbone" not in name], "lr": args.lr*1}],
                              lr=args.lr, weight_decay=args.weight_decay)
        self.model = self.model.cuda()

        self.iters = 0
        self.total_iters = len(self.trainloader) * args.epochs
        self.previous_best = 0.0
        self.seg_best = 0.0
        self.change_best = 0.0

    def training(self, epoch):
        curr_epoch = epoch
        tbar = tqdm(self.trainloader)
        self.model.train()
        total_loss = 0.0
        total_loss_seg = 0.0
        total_loss_bn = 0.0
        total_loss_similarity = 0.0
        ##
        total_loss_kd = 0.0
        total_loss_sim = 0.0

        curr_iter = curr_epoch * len(self.trainloader)

        for i, (img1, img2, depth1, depth2, mask1, mask2, mask_bn, id) in enumerate(tbar):
            running_iter = curr_iter + i + 1
            img1, img2 = img1.cuda(), img2.cuda()
            depth1, depth2 = depth1.cuda(), depth2.cuda()
            mask1, mask2, mask_bn = mask1.cuda().long(), mask2.cuda().long(), mask_bn.cuda().long()
            # out1, out2, out_bn, loss_sim = self.model(img1, img2, depth1, depth2)    #[b,6,512,512],[b,6,512,512],[b,512,512]
            
            # loss1 = self.criterion_seg(out1, mask1 - 1)
            # loss2 = self.criterion_seg(out2, mask2 - 1)
            # loss_seg = loss1 * 0.5 + loss2 * 0.5
            # loss_similarity = self.criterion_sc(out1[:, 0:], out2[:, 0:], mask_bn)
            # loss_bn_1 = self.criterion_bn(out_bn, mask_bn)
            # loss_bn_1[mask_bn == 1] *= 2
            # loss_bn_1 = loss_bn_1.mean()
            
            # prob_bn = torch.softmax(out_bn, dim=1)[:, 1, :, :]   # 取变化类概率
            # loss_bn_2 = self.criterion_bn_2(prob_bn, mask_bn.float())
            # loss_bn = loss_bn_1 + loss_bn_2
            # ##
            # # loss_kd = self.kd(out1, out2, out_bn)
            # loss_kd = self.kd(out1, out2, out_bn, depth1, depth2)
            # loss = loss_bn + loss_seg + loss_similarity + 0.5*(loss_kd + loss_sim)

            out1, out2, out_bn, loss_sim = self.model(img1, img2, depth1, depth2)

            target1 = mask1 - 1
            target2 = mask2 - 1

            valid1 = target1 != -1
            valid2 = target2 != -1

            loss1 = self.criterion_seg(out1, target1) if valid1.any() else out1.sum() * 0.0
            loss2 = self.criterion_seg(out2, target2) if valid2.any() else out2.sum() * 0.0

            loss_seg = loss1 * 0.5 + loss2 * 0.5

            loss_similarity = self.criterion_sc(out1[:, 0:], out2[:, 0:], mask_bn)

            loss_bn_1 = self.criterion_bn(out_bn, mask_bn)
            loss_bn_1[mask_bn == 1] *= 2
            loss_bn_1 = loss_bn_1.mean()

            prob_bn = torch.softmax(out_bn, dim=1)[:, 1, :, :]
            loss_bn_2 = self.criterion_bn_2(prob_bn, mask_bn.float())
            loss_bn = loss_bn_1 + loss_bn_2

            loss_kd = self.kd(out1, out2, out_bn, depth1, depth2)
            loss = loss_bn + loss_seg + loss_similarity + 0.5 * (loss_kd + loss_sim)

            if not torch.isfinite(loss):
                print("Non-finite loss detected, skip this batch.")
                print("loss:", loss)
                print("loss_seg:", loss_seg)
                print("loss_bn:", loss_bn)
                print("loss_similarity:", loss_similarity)
                print("loss_kd:", loss_kd)
                print("loss_sim:", loss_sim)
                print("mask1 unique:", torch.unique(mask1))
                print("mask2 unique:", torch.unique(mask2))
                print("mask_bn unique:", torch.unique(mask_bn))
                continue

            total_loss_seg += loss_seg.item()
            total_loss_similarity += loss_similarity.item()
            total_loss_bn += loss_bn.item()
            total_loss_sim += loss_sim.item()
            total_loss_kd += loss_kd.item()
            total_loss += loss.item()

            self.iters += 1
            if args.warmup:
                warmup_steps = len(self.trainloader) * (args.epochs/5)
                if warmup_steps and self.iters < warmup_steps:
                    warmup_percent_done = self.iters / warmup_steps
                    lr = args.lr * warmup_percent_done
                else:
                    lr = self.args.lr * (1. - float(self.iters) / self.total_iters) ** 1.5
            else:
                lr = self.args.lr * (1. - float(self.iters) / self.total_iters) ** 1.5

            self.optimizer.param_groups[0]["lr"] = lr
            if args.pretrain_from:
                self.optimizer.param_groups[1]["lr"] = lr * 1.0
            else:
                self.optimizer.param_groups[1]["lr"] = lr * 1.0

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()           
            
            tbar.set_description("Loss: %.3f, Semantic Loss: %.3f, Binary Loss: %.3f, Similarity Loss: %.3f" %
                                 (total_loss / (i + 1), total_loss_seg / (i + 1), total_loss_bn / (i + 1), total_loss_similarity / (i + 1)))

            self.writer.add_scalar('train total_loss', total_loss / (i + 1), running_iter)
            self.writer.add_scalar('train seg_loss', total_loss_seg / (i + 1), running_iter)
            self.writer.add_scalar('train bn_loss', total_loss_bn / (i + 1), running_iter)
            self.writer.add_scalar('train sc_loss', total_loss_similarity / (i + 1), running_iter)
            self.writer.add_scalar('lr', self.optimizer.param_groups[0]["lr"], running_iter)

    def validation(self, epoch):
        curr_epoch = epoch
        tbar = tqdm(self.valloader)
        self.model.eval()
        metric = IOUandSek(num_classes=len(ChangeDetection_FZSCD.CLASSES))
        if self.args.save_mask:
            cmap = color_map()

        with torch.no_grad():
            for img1, img2, depth1, depth2, mask1, mask2, mask_bn, _ in tbar:
                img1, img2 = img1.cuda(), img2.cuda()
                depth1, depth2 = depth1.cuda(), depth2.cuda()

                out1, out2, out_bn, _ = self.model(img1, img2, depth1, depth2)
                out1 = torch.argmax(out1, dim=1).cpu().numpy() + 1
                out2 = torch.argmax(out2, dim=1).cpu().numpy() + 1
                # out_bn = (out_bn > 0.5).cpu().numpy().astype(np.uint8)
                out_bn = torch.argmax(out_bn, dim=1).cpu().numpy().astype(np.uint8)
                out1[out_bn == 0] = 0
                out2[out_bn == 0] = 0

                metric.add_batch(out1, mask1.numpy())
                metric.add_batch(out2, mask2.numpy())
                change_ratio, score, miou, sek, Fscd, OA, SC_Precision, SC_Recall = metric.evaluate_SECOND()

                tbar.set_description(
                    "miou: %.4f, sek: %.4f, score: %.4f, Fscd: %.4f, OA: %.4f, SC_Precision: %.4f, SC_Recall: %.4f" % (
                    miou, sek, score, Fscd, OA, SC_Precision, SC_Recall))

        if score >= self.previous_best:
            model_path = "checkpoints/%s/%s/%s" % \
                         (self.args.data_name, self.args.Net_name, self.args.backbone)
            if not os.path.exists(model_path): os.makedirs(model_path)
            torch.save(self.model.state_dict(), "checkpoints/%s/%s/%s/epoch%i_Score%.2f_mIOU%.2f_Sek%.2f_Fscd%.2f_OA%.2f.pth" %
                (self.args.data_name, self.args.Net_name, self.args.backbone, curr_epoch, score*100, miou*100, sek*100, Fscd*100, OA*100))

            self.previous_best = score

        self.writer.add_scalar('val_Score', score, curr_epoch)
        self.writer.add_scalar('val_mIOU', miou, curr_epoch)
        self.writer.add_scalar('val_Sek', sek, curr_epoch)
        self.writer.add_scalar('val_Fscd', Fscd, curr_epoch)
        self.writer.add_scalar('val_OA', OA, curr_epoch)


if __name__ == "__main__":
    args = Options().parse()
    trainer = Trainer(args)

    if args.load_from:
        trainer.validation()

    for epoch in range(args.epochs):
        print("\n==> Epoches %i, learning rate = %.5f\t\t\t\t previous best = %.5f" %
              (epoch, trainer.optimizer.param_groups[0]["lr"], trainer.previous_best))
        trainer.training(epoch)
        trainer.validation(epoch)
