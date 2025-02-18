import numpy as np
import random
import torch
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.models.sr_model import SRModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
import torch.nn as nn
from torch.nn import functional as F

class DynamicConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(DynamicConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # 动态卷积核生成器
        self.kernel_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化，提取全局特征
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),  # 压缩通道数
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, out_channels * in_channels * kernel_size * kernel_size, kernel_size=1)  # 生成卷积核
        )

        # 是否使用偏置
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        batch_size, _, height, width = x.size()

        # 生成动态卷积核
        kernel_weights = self.kernel_generator(x)  # [batch_size, out_channels * in_channels * k * k, 1, 1]
        kernel_weights = kernel_weights.view(batch_size, self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)

        # 对每个样本应用动态卷积
        output = []
        for i in range(batch_size):
            dynamic_kernel = kernel_weights[i]  # [out_channels, in_channels, kernel_size, kernel_size]
            input_sample = x[i].unsqueeze(0)  # [1, in_channels, height, width]
            output_sample = F.conv2d(input_sample, dynamic_kernel, bias=self.bias, stride=self.stride,
                                     padding=self.padding, dilation=self.dilation, groups=self.groups)
            output.append(output_sample)

        return torch.cat(output, dim=0)

@MODEL_REGISTRY.register()
class AMHDMSRNetModel(SRModel):

    def __init__(self, opt):
        super(AMHDMSRNetModel, self).__init__(opt)
        self.jpeger = DiffJPEG(differentiable=False).cuda()  # 模拟 JPEG 压缩伪影
        self.usm_sharpener = USMSharp().cuda()  # 执行 USM 锐化
        self.queue_size = opt.get('queue_size', 180)

        # 动态卷积模块
        self.dynamic_conv = DynamicConv2D(in_channels=3, out_channels=3, kernel_size=7).cuda()

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # 队列已满
            idx = torch.randperm(self.queue_size)  # 随机打乱
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()
            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def feed_data(self, data):
        if self.is_train and self.opt.get('high_order_degradation', True):
            # 训练数据合成
            self.gt = data['gt'].to(self.device)
            if self.opt['gt_usm'] is True:
                self.gt = self.usm_sharpener(self.gt)  # 对 GT 图像进行 USM 锐化

            ori_h, ori_w = self.gt.size()[2:4]

            # ----------------------- 第一次退化过程 ----------------------- #
            # 动态卷积模糊
            blurred = self.dynamic_conv(self.gt)

            # 随机失真
            distorted = self._apply_distortion(blurred)

            # 随机亮度变化
            brightness_adjusted = self._adjust_brightness(distorted)

            # 随机旋转
            rotated = self._random_rotate(brightness_adjusted)

            # 随机缩放
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
            scale = {
                'up': lambda: np.random.uniform(1, self.opt['resize_range'][1]),
                'down': lambda: np.random.uniform(self.opt['resize_range'][0], 1),
                'keep': lambda: 1
            }[updown_type]()
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            resized = F.interpolate(rotated, scale_factor=scale, mode=mode)

            # 下采样
            downsampled = self._downsample(resized)

            # 动态噪声生成
            gray_noise_prob = self.opt['gray_noise_prob']
            if np.random.uniform() < self.opt['gaussian_noise_prob']:
                noise_level = self._dynamic_noise_level(downsampled)  # 动态生成噪声水平
                downsampled = random_add_gaussian_noise_pt(
                    downsampled, sigma_range=noise_level, clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                noise_scale = self._dynamic_noise_scale(downsampled)  # 动态生成泊松噪声尺度
                downsampled = random_add_poisson_noise_pt(
                    downsampled, scale_range=noise_scale, gray_prob=gray_noise_prob, clip=True, rounds=False)


            jpeg_p = downsampled.new_zeros(downsampled.size(0)).uniform_(*self.opt['jpeg_range'])
            downsampled = torch.clamp(downsampled, 0, 1)  # 限制范围 [0, 1]
            downsampled = self.jpeger(downsampled, quality=jpeg_p)

            # ----------------------- 第二次退化过程（随机控制） ----------------------- #
            if np.random.uniform() < 0.5:  # 随机控制是否进行第二次退化
                # 动态卷积模糊
                second_blurred = self.dynamic_conv(downsampled)
                if np.random.uniform() < self.opt['second_blur_prob']:
                    second_blurred = self.dynamic_conv(second_blurred)

                # 随机失真
                distorted = self._apply_distortion(second_blurred)

                # 随机亮度变化
                brightness_adjusted = self._adjust_brightness(distorted)

                # 随机旋转
                rotated = self._random_rotate(brightness_adjusted)

                # 再次随机缩放
                updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
                scale = {
                    'up': lambda: np.random.uniform(1, self.opt['resize_range2'][1]),
                    'down': lambda: np.random.uniform(self.opt['resize_range2'][0], 1),
                    'keep': lambda: 1
                }[updown_type]()
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                final_resized = F.interpolate(
                    rotated, size=(int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)), mode=mode)

                # 下采样
                downsampled = self._downsample(final_resized)

                # 动态噪声生成
                gray_noise_prob = self.opt['gray_noise_prob2']
                if np.random.uniform() < self.opt['gaussian_noise_prob2']:
                    noise_level = self._dynamic_noise_level(downsampled)
                    downsampled = random_add_gaussian_noise_pt(
                        downsampled, sigma_range=noise_level, clip=True, rounds=False, gray_prob=gray_noise_prob)
                else:
                    noise_scale = self._dynamic_noise_scale(downsampled)
                    downsampled = random_add_poisson_noise_pt(
                        downsampled, scale_range=noise_scale, gray_prob=gray_noise_prob, clip=True, rounds=False)


                if np.random.uniform() < 0.5:
                    mode = random.choice(['area', 'bilinear', 'bicubic'])
                    downsampled = F.interpolate(downsampled, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                    downsampled = filter2D(downsampled, self.sinc_kernel)
                    jpeg_p = downsampled.new_zeros(downsampled.size(0)).uniform_(*self.opt['jpeg_range2'])
                    downsampled = torch.clamp(downsampled, 0, 1)
                    downsampled = self.jpeger(downsampled, quality=jpeg_p)
                else:
                    jpeg_p = downsampled.new_zeros(downsampled.size(0)).uniform_(*self.opt['jpeg_range2'])
                    downsampled = torch.clamp(downsampled, 0, 1)
                    downsampled = self.jpeger(downsampled, quality=jpeg_p)
                    mode = random.choice(['area', 'bilinear', 'bicubic'])
                    downsampled = F.interpolate(downsampled, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                    downsampled = filter2D(downsampled, self.sinc_kernel)
            else:
                downsampled = downsampled  # 如果不进行第二次退化，则直接使用第一次退化的结果

            # 后处理
            self.lq = torch.clamp((downsampled * 255.0).round(), 0, 255) / 255.
            gt_size = self.opt['gt_size']
            self.gt, self.lq = paired_random_crop(self.gt, self.lq, gt_size, self.opt['scale'])

            # 更新队列
            self._dequeue_and_enqueue()
            self.lq = self.lq.contiguous()

        else:
            # 非训练模式
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)
                self.gt_usm = self.usm_sharpener(self.gt)

    def _apply_distortion(self, image):
        """
        应用随机失真。
        """
        distortion_strength = np.random.uniform(0.9, 1.1)  # 随机强度
        return image * distortion_strength

    def _adjust_brightness(self, image):
        """
        调整亮度。
        """
        brightness_factor = np.random.uniform(0.8, 1.2)  # 随机亮度因子
        return torch.clamp(image * brightness_factor, 0, 1)

    def _random_rotate(self, image):
        """
        随机旋转图像。
        """
        angle = np.random.uniform(-10, 10)  # 随机角度（-10° 到 10°）
        return F.rotate(image, angle)

    def _downsample(self, image):
        """
        下采样图像。
        """
        scale_factor = np.random.uniform(0.5, 1.0)  # 随机下采样比例
        return F.interpolate(image, scale_factor=scale_factor, mode='bicubic')

    def _dynamic_noise_level(self, image):
        """
        根据图像内容动态生成高斯噪声水平。
        """
        noise_level = torch.std(image, dim=[2, 3], keepdim=True) * 0.1  # 动态调整噪声强度
        return noise_level.squeeze()

    def _dynamic_noise_scale(self, image):
        """
        根据图像内容动态生成泊松噪声尺度。
        """
        noise_scale = torch.mean(image, dim=[2, 3], keepdim=True) * 0.2  # 动态调整噪声尺度
        return noise_scale.squeeze()

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # 验证时不使用合成过程
        self.is_train = False
        super(AMHDMSRNetModel, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True
