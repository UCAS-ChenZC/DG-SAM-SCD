from datasets.augmentation_depth import augmentation_compose, augmentation_compose_BCD
import numpy as np
import os
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from osgeo import gdal

def read_image(image_path):
    """
    高效读取单波段灰度图、RGB影像或四波段遥感影像（RGB+NIR）。

    参数:
    image_path (str): 影像文件的路径。

    返回:
    img_data (numpy.ndarray): 影像数据。
    bands (list): 波段名称列表。
    """
    # 注册所有驱动
    gdal.AllRegister()
    # 打开影像文件
    dataset = gdal.Open(image_path, gdal.GA_ReadOnly)
    if not dataset:
        raise IOError("无法打开文件")
    # 获取影像的基本信息
    nBandCount = dataset.RasterCount  # 波段数
    nRows = dataset.RasterYSize  # 影像的高度（像元数目），行数
    nCols = dataset.RasterXSize  # 影像的宽度（像元数目），列数
    # 读取所有波段的数据
    img_data = dataset.ReadAsArray().astype('float32')  # 读取所有波段数据并转换为float32类型
    # 根据波段数量确定波段名称
    if nBandCount == 1:  # 灰度图
        bands = ['Gray']
    elif nBandCount == 3:  # RGB影像
        bands = ['Red', 'Green', 'Blue']
    elif nBandCount == 4:  # 四波段遥感影像（RGB+NIR）
        bands = ['Red', 'Green', 'Blue', 'NIR']
    else:
        raise ValueError("Unsupported number of bands. Only single band (grayscale), RGB, or RGB+NIR are supported.")
    # 关闭数据集
    dataset = None
    return img_data

class ChangeDetection_SECOND(Dataset):
    #SECOND
    CLASSES = ['未变化区域', '水体', '地面', '低矮植被', '树木', '建筑物', '运动场']
    # FZSCD
    # CLASSES = ['未变化区域', '水体', '地面', '植被', '建筑物']
    # WUSU
    # CLASSES = ['road','low-building', 'high-building', 'Arable', 'woodland', 'grassland', 'river', 'lake', 'structure', 'excavation', 'bare']

    def __init__(self, root, mode, use_pseudo_label=False):
        super(ChangeDetection_SECOND, self).__init__()
        self.root = root
        self.mode = mode

        if mode == 'train':
            self.root = os.path.join(self.root, 'train')
            self.ids = os.listdir(os.path.join(self.root, "im1"))
            self.ids.sort()
        elif mode == 'val':
            self.root = os.path.join(self.root, 'val')
            self.ids = os.listdir(os.path.join(self.root, 'im1'))
            self.ids.sort()
        elif mode == 'test':
            self.root = os.path.join(self.root, 'test')
            self.ids = os.listdir(os.path.join(self.root, 'im1'))
            self.ids.sort()

        self.transform = augmentation_compose
        self.normalize = transforms.Compose([
            transforms.ToTensor(),      #这scale至[0,1]
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __getitem__(self, index):
        id = self.ids[index]

        img1 = np.array(Image.open(os.path.join(self.root, 'im1', id)))
        img2 = np.array(Image.open(os.path.join(self.root, 'im2', id)))
        depth1 = np.array(Image.open(os.path.join(self.root, 'depth_A', id)))
        depth2 = np.array(Image.open(os.path.join(self.root, 'depth_B', id)))
        mask1 = np.array(Image.open(os.path.join(self.root, 'label1', id)))
        mask2 = np.array(Image.open(os.path.join(self.root, 'label2', id)))

        mask_bin = np.zeros_like(mask1)
        mask_bin[mask1 != 0 ] = 1

        if self.mode == 'train':
            sample = self.transform({'img1': img1, 'img2': img2, 'depth1': depth1, 'depth2': depth2, 'mask1': mask1, 'mask2': mask2,
                                        'gt_mask': mask_bin})
            img1, img2, depth1, depth2, mask1, mask2, mask_bin = sample['img1'], sample['img2'], sample['depth1'], \
                                                                    sample['depth2'], sample['mask1'], sample['mask2'], sample['gt_mask']
        img1 = self.normalize(img1)
        img2 = self.normalize(img2)
        
        # depth 转 tensor
        depth1 = torch.from_numpy(depth1).float().unsqueeze(0)
        depth2 = torch.from_numpy(depth2).float().unsqueeze(0)
        # 可选：归一化到 [0,1]
        if depth1.max() > 1:
            depth1 = depth1 / 255.0
        if depth2.max() > 1:
            depth2 = depth2 / 255.0
            
        mask1 = torch.from_numpy(np.array(mask1)).long()
        mask2 = torch.from_numpy(np.array(mask2)).long()
        mask_bin = torch.from_numpy(np.array(mask_bin)).float()

        return img1, img2, depth1, depth2, mask1, mask2, mask_bin, id

    def __len__(self):
        return len(self.ids)



class ChangeDetection_FZSCD(Dataset):

    # FZSCD
    CLASSES = ['未变化区域', '裸地', '建筑物', '植被', '水体', '道路', '其他']
    def __init__(self, root, mode, use_pseudo_label=False):
        super(ChangeDetection_FZSCD, self).__init__()
        self.root = root
        self.mode = mode

        if mode == 'train':
            self.root = os.path.join(self.root, 'train')
            self.ids = os.listdir(os.path.join(self.root, "im1"))
            self.ids.sort()
        elif mode == 'val':
            self.root = os.path.join(self.root, 'val')
            self.ids = os.listdir(os.path.join(self.root, 'im1'))
            self.ids.sort()
        elif mode == 'test':
            self.root = os.path.join(self.root, 'test')
            self.ids = os.listdir(os.path.join(self.root, 'im1'))
            self.ids.sort()

        self.transform = augmentation_compose
        self.normalize = transforms.Compose([
            transforms.ToTensor(),      #这scale至[0,1]
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __getitem__(self, index):
        id = self.ids[index]
        img1 = np.array(Image.open(os.path.join(self.root, 'im1', id)))
        img2 = np.array(Image.open(os.path.join(self.root, 'im2', id).replace('T1m','T2')))
        depth1 = np.array(Image.open(os.path.join(self.root, 'depth_A', id)))
        depth2 = np.array(Image.open(os.path.join(self.root, 'depth_B', id)))
        mask1 = np.array(Image.open(os.path.join(self.root, 'label1', id).replace('T1m','T1_label')))
        mask2 = np.array(Image.open(os.path.join(self.root, 'label2', id).replace('T1m','T2_label')))
        mask1[mask1 == 15] = 0
        mask2[mask2 == 15] = 0
        mask_bin = np.zeros_like(mask1)
        mask_bin[mask1 != 0] = 1

        if self.mode == 'train':
            sample = self.transform({'img1': img1, 'img2': img2, 'depth1': depth1, 'depth2': depth2, 'mask1': mask1, 'mask2': mask2,
                                        'gt_mask': mask_bin})
            img1, img2, depth1, depth2, mask1, mask2, mask_bin = sample['img1'], sample['img2'], sample['depth1'], \
                                                                    sample['depth2'], sample['mask1'], sample['mask2'], sample['gt_mask']
            # sample = self.transform({'img1': img1, 'img2': img2, 'mask1': mask1, 'mask2': mask2,
            #                             'gt_mask': mask_bin})
            # img1, img2, mask1, mask2, mask_bin = sample['img1'], sample['img2'], sample['mask1'], \
            #                                         sample['mask2'], sample['gt_mask']
        img1 = self.normalize(img1)
        img2 = self.normalize(img2)
        
        # depth 转 tensor
        depth1 = torch.from_numpy(depth1).float().unsqueeze(0)
        depth2 = torch.from_numpy(depth2).float().unsqueeze(0)
        # 可选：归一化到 [0,1]
        if depth1.max() > 1:
            depth1 = depth1 / 255.0
        if depth2.max() > 1:
            depth2 = depth2 / 255.0

        mask1 = torch.from_numpy(np.array(mask1)).long()
        mask2 = torch.from_numpy(np.array(mask2)).long()
        mask_bin = torch.from_numpy(np.array(mask_bin)).float()

        return img1, img2, depth1, depth2, mask1, mask2, mask_bin, id

    def __len__(self):
        return len(self.ids)
class ChangeDetection_WUSU(Dataset):
    # WUSU
    CLASSES = ['nochange', 'Road', 'Low building', 'High building', 'ArableLand', 'Woodland', 'Grassland', 'water', 'lake', 'structure', 'excavation', 'bare']

    def __init__(self, root, mode, use_pseudo_label=False):
        super(ChangeDetection_WUSU, self).__init__()
        self.root = root
        self.mode = mode

        if mode == 'train':
            self.root = os.path.join(self.root, 'train')
            self.ids = os.listdir(os.path.join(self.root, "im1"))
            self.ids.sort()
        elif mode == 'val':
            self.root = os.path.join(self.root, 'val')
            self.ids = os.listdir(os.path.join(self.root, 'im1'))
            self.ids.sort()
        elif mode == 'test':
            self.root = os.path.join(self.root, 'test')
            self.ids = os.listdir(os.path.join(self.root, 'im1'))
            self.ids.sort()

        self.transform = augmentation_compose
        self.normalize = transforms.Compose([
            transforms.ToTensor(),      #这scale至[0,1]
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __getitem__(self, index):
        id = self.ids[index]

        img1 = np.array(read_image(os.path.join(self.root, 'im1', id)))
        img2 = np.array(read_image(os.path.join(self.root, 'im2', id)))
        img3 = np.array(read_image(os.path.join(self.root, 'im3', id)))
        
        depth1 = np.array(Image.open(os.path.join(self.root, 'depth_1', id)))
        depth2 = np.array(Image.open(os.path.join(self.root, 'depth_2', id)))
        depth3 = np.array(Image.open(os.path.join(self.root, 'depth_3', id)))

        mask1 = np.array(Image.open(os.path.join(self.root, 'label1', id)))
        mask2 = np.array(Image.open(os.path.join(self.root, 'label2', id)))
        mask3 = np.array(Image.open(os.path.join(self.root, 'label3', id)))
        mask_bin = mask2 - mask1
        mask_bin[mask_bin != 0 ] = 1


        if self.mode == 'train':
            sample = self.transform({'img1': img1, 'img2': img2, 'img3': img3, 'mask1': mask1, 'mask2': mask2, 'mask3': mask3,
                                        'gt_mask': mask_bin})
            img1, img2, img3, mask1, mask2, mask3, mask_bin = sample['img1'], sample['img2'], sample['img3'], sample['mask1'], \
                                                                    sample['mask2'], sample['mask3'], sample['gt_mask']
        img1 = self.normalize(img1)
        img2 = self.normalize(img2)
        img3 = self.normalize(img3)
        
        # depth 转 tensor
        depth1 = torch.from_numpy(depth1).float().unsqueeze(0)
        depth2 = torch.from_numpy(depth2).float().unsqueeze(0)
        depth3 = torch.from_numpy(depth3).float().unsqueeze(0)
        # 可选：归一化到 [0,1]
        if depth1.max() > 1:
            depth1 = depth1 / 255.0
        if depth2.max() > 1:
            depth2 = depth2 / 255.0
        if depth3.max() > 1:
            depth3 = depth3 / 255.0

        mask1 = torch.from_numpy(np.array(mask1)).long()
        mask2 = torch.from_numpy(np.array(mask2)).long()
        mask3 = torch.from_numpy(np.array(mask3)).long()
        mask_bin = torch.from_numpy(np.array(mask_bin)).float()

        return img1, img2, img3, mask1, mask2, mask3, mask_bin, id

    def __len__(self):
        return len(self.ids)




class ChangeDetection_Landsat_SCD(Dataset):
    CLASSES = ['未变化区域', '农田', '沙漠', '建筑物', '水体']
    def __init__(self, root, mode, use_pseudo_label=False):
        super(ChangeDetection_Landsat_SCD, self).__init__()
        self.root = root

        self.mode = mode

        if mode == 'train':
            self.root = os.path.join(self.root, 'train')
            self.ids = os.listdir(os.path.join(self.root, "A"))
            self.ids.sort()
        elif mode == 'val':
            self.root = os.path.join(self.root, 'val')
            self.ids = os.listdir(os.path.join(self.root, 'A'))
            self.ids.sort()
        elif mode == 'test':
            self.root = os.path.join(self.root, 'test')
            self.ids = os.listdir(os.path.join(self.root, 'A'))
            self.ids.sort()

        self.transform = augmentation_compose
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __getitem__(self, index):
        id = self.ids[index]

        img1 = np.array(Image.open(os.path.join(self.root, 'A', id)))
        img2 = np.array(Image.open(os.path.join(self.root, 'B', id)))
        depth1 = np.array(Image.open(os.path.join(self.root, 'depth_A', id)))
        depth2 = np.array(Image.open(os.path.join(self.root, 'depth_B', id)))
        mask1 = np.array(Image.open(os.path.join(self.root, 'label1', id)))
        mask2 = np.array(Image.open(os.path.join(self.root, 'label2', id)))

        mask_bin = np.zeros_like(mask1)
        mask_bin[mask1 != 0] = 1

        img1 = self.normalize(img1)
        img2 = self.normalize(img2)
        
        # depth 转 tensor
        depth1 = torch.from_numpy(depth1).float().unsqueeze(0)
        depth2 = torch.from_numpy(depth2).float().unsqueeze(0)
        # 可选：归一化到 [0,1]
        if depth1.max() > 1:
            depth1 = depth1 / 255.0
        if depth2.max() > 1:
            depth2 = depth2 / 255.0
            
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_edge = cv2.GaussianBlur(mask_bin * 255, (3, 3), 0)
        mask_edge = cv2.Canny(mask_edge, 50, 150)
        mask_edge = cv2.dilate(mask_edge, kernel, iterations=2)
        mask_edge = torch.from_numpy(np.array(mask_edge) // 255).long()

        mask1 = torch.from_numpy(np.array(mask1)).long()
        mask2 = torch.from_numpy(np.array(mask2)).long()
        mask_bin = torch.from_numpy(np.array(mask_bin)).float()

        return img1, img2, depth1, depth2, mask1, mask2, mask_bin, id

    def __len__(self):
        return len(self.ids)

    def __len__(self):
        return len(self.ids)


class ChangeDetection_LEVIR_CD(Dataset):
    #SECOND
    CLASSES = ['未变化区域', '变化']
    def __init__(self, root, mode, use_pseudo_label=False):
        super(ChangeDetection_LEVIR_CD, self).__init__()
        self.root = root
        self.mode = mode

        if mode == 'train':
            self.root = os.path.join(self.root, 'train')
            self.ids = os.listdir(os.path.join(self.root, "im1"))
            self.ids.sort()
        elif mode == 'val':
            self.root = os.path.join(self.root, 'val')
            self.ids = os.listdir(os.path.join(self.root, 'im1'))
            self.ids.sort()
        elif mode == 'test':
            self.root = os.path.join(self.root, 'test')
            self.ids = os.listdir(os.path.join(self.root, 'im1'))
            self.ids.sort()

        self.transform = augmentation_compose_BCD
        self.normalize = transforms.Compose([
            transforms.ToTensor(),      #这scale至[0,1]
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            # transforms.Normalize((123.675, 116.28, 103.53), (58.395, 57.12, 57.375))
        ])

    def __getitem__(self, index):
        id = self.ids[index]

        img1 = np.array(Image.open(os.path.join(self.root, 'im1', id)))
        img2 = np.array(Image.open(os.path.join(self.root, 'im2', id)))
        mask_bin = np.array(Image.open(os.path.join(self.root, 'label', id)))
        mask_bin[mask_bin == 255] = 1
        mask_bin[mask_bin != 1] = 0
        if self.mode == 'train':
            sample = self.transform({'img1': img1, 'img2': img2, 'gt_mask': mask_bin})
            img1, img2, mask_bin = sample['img1'], sample['img2'], sample['gt_mask']
        img1 = self.normalize(img1)
        img2 = self.normalize(img2)
        mask_bin = torch.from_numpy(np.array(mask_bin)).float()

        # print(mask_bin)
        return img1, img2, mask_bin, id

    def __len__(self):
        return len(self.ids)
    
class ChangeDetection_JL1H(Dataset):
    
    CLASSES = ['unchanged', 'farm', 'road', 'tree', 'building', 'other']

    def __init__(self, root, mode, use_pseudo_label=False):
        super(ChangeDetection_JL1H, self).__init__()
        self.root = root
        self.mode = mode

        if mode == 'train':
            self.root = os.path.join(self.root, 'train')
            self.ids = os.listdir(os.path.join(self.root, "im1_png"))
            self.ids.sort()
        elif mode == 'val':
            self.root = os.path.join(self.root, 'val')
            self.ids = os.listdir(os.path.join(self.root, 'im1_png'))
            self.ids.sort()
        elif mode == 'test':
            self.root = os.path.join(self.root, 'test')
            self.ids = os.listdir(os.path.join(self.root, 'im1_png'))
            self.ids.sort()

        self.transform = augmentation_compose
        self.normalize = transforms.Compose([
            transforms.ToTensor(),      #这scale至[0,1]
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __getitem__(self, index):
        id = self.ids[index]

        img1 = np.array(Image.open(os.path.join(self.root, 'im1_png', id)))
        img2 = np.array(Image.open(os.path.join(self.root, 'im2_png', id)))
        depth1 = np.array(Image.open(os.path.join(self.root, 'depth_A', id)))
        depth2 = np.array(Image.open(os.path.join(self.root, 'depth_B', id)))
        mask1 = np.array(Image.open(os.path.join(self.root, 'label1_gray', id)))
        mask2 = np.array(Image.open(os.path.join(self.root, 'label2_gray', id)))

        mask_bin = np.zeros_like(mask1)
        mask_bin[mask1 != 0 ] = 1

        if self.mode == 'train':
            sample = self.transform({'img1': img1, 'img2': img2, 'depth1': depth1, 'depth2': depth2, 'mask1': mask1, 'mask2': mask2,
                                        'gt_mask': mask_bin})
            img1, img2, depth1, depth2, mask1, mask2, mask_bin = sample['img1'], sample['img2'], sample['depth1'], \
                                                                    sample['depth2'], sample['mask1'], sample['mask2'], sample['gt_mask']
        img1 = self.normalize(img1)
        img2 = self.normalize(img2)
        
        # depth 转 tensor
        depth1 = torch.from_numpy(depth1).float().unsqueeze(0)
        depth2 = torch.from_numpy(depth2).float().unsqueeze(0)
        # 可选：归一化到 [0,1]
        if depth1.max() > 1:
            depth1 = depth1 / 255.0
        if depth2.max() > 1:
            depth2 = depth2 / 255.0
            
        mask1 = torch.from_numpy(np.array(mask1)).long()
        mask2 = torch.from_numpy(np.array(mask2)).long()
        mask_bin = torch.from_numpy(np.array(mask_bin)).float()

        return img1, img2, depth1, depth2, mask1, mask2, mask_bin, id

    def __len__(self):
        return len(self.ids)