import numpy as np
import random
import torch
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.models.srgan_model import SRGANModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from collections import OrderedDict
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

        # 定义一个小型网络来生成动态卷积核
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
class AMHDMSRGANModel(SRGANModel):

    def __init__(self, opt):
        super(AMHDMSRGANModel, self).__init__(opt)
        self.jpeger = DiffJPEG(differentiable=False).cuda()  # 模拟 JPEG 压缩伪影
        self.usm_sharpener = USMSharp().cuda()  # 执行 USM 锐化
        self.queue_size = opt.get('queue_size', 180)

        # 动态卷积模块
        self.dynamic_conv = DynamicConv2D(in_channels=3, out_channels=3, kernel_size=7).cuda()

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        # 初始化队列逻辑保持不变
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
            self.gt_usm = self.usm_sharpener(self.gt)

            ori_h, ori_w = self.gt.size()[2:4]

            # ----------------------- 第一次退化过程 ----------------------- #
            # 动态卷积模糊
            blurred = self.dynamic_conv(self.gt_usm)

            # 随机缩放
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['resize_range'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['resize_range'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            resized = F.interpolate(blurred, scale_factor=scale, mode=mode)

            # 添加失真、亮度变化、旋转、下采样
            resized = self.apply_distortion(resized, distortion_strength=np.random.uniform(0.05, 0.2))
            resized = self.adjust_brightness(resized)
            resized = self.rotate_image(resized)
            resized = self.downsample_image(resized)

            # 动态噪声生成
            gray_noise_prob = self.opt['gray_noise_prob']
            if np.random.uniform() < self.opt['gaussian_noise_prob']:
                noise_level = self._dynamic_noise_level(resized)  # 动态生成噪声水平
                resized = random_add_gaussian_noise_pt(
                    resized, sigma_range=noise_level, clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                noise_scale = self._dynamic_noise_scale(resized)  # 动态生成泊松噪声尺度
                resized = random_add_poisson_noise_pt(
                    resized, scale_range=noise_scale, gray_prob=gray_noise_prob, clip=True, rounds=False)


            jpeg_p = resized.new_zeros(resized.size(0)).uniform_(*self.opt['jpeg_range'])
            resized = torch.clamp(resized, 0, 1)  # 限制范围 [0, 1]
            resized = self.jpeger(resized, quality=jpeg_p)

            # ----------------------- 第二次退化过程 ----------------------- #
            if np.random.uniform() < 0.5:
                final_resized = resized  # 跳过第二次退化
            else:
                # 动态卷积模糊
                second_blurred = self.dynamic_conv(resized)
                if np.random.uniform() < self.opt['second_blur_prob']:
                    second_blurred = self.dynamic_conv(second_blurred)

                # 再次随机缩放
                updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
                if updown_type == 'up':
                    scale = np.random.uniform(1, self.opt['resize_range2'][1])
                elif updown_type == 'down':
                    scale = np.random.uniform(self.opt['resize_range2'][0], 1)
                else:
                    scale = 1
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                final_resized = F.interpolate(
                    second_blurred, size=(int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)), mode=mode)

                # 添加失真、亮度变化、旋转、下采样
                final_resized = self.apply_distortion(final_resized, distortion_strength=np.random.uniform(0.05, 0.2))
                final_resized = self.adjust_brightness(final_resized)
                final_resized = self.rotate_image(final_resized)
                final_resized = self.downsample_image(final_resized)

                # 动态噪声生成
                gray_noise_prob = self.opt['gray_noise_prob2']
                if np.random.uniform() < self.opt['gaussian_noise_prob2']:
                    noise_level = self._dynamic_noise_level(final_resized)
                    final_resized = random_add_gaussian_noise_pt(
                        final_resized, sigma_range=noise_level, clip=True, rounds=False, gray_prob=gray_noise_prob)
                else:
                    noise_scale = self._dynamic_noise_scale(final_resized)
                    final_resized = random_add_poisson_noise_pt(
                        final_resized, scale_range=noise_scale, gray_prob=gray_noise_prob, clip=True, rounds=False)


                if np.random.uniform() < 0.5:
                    mode = random.choice(['area', 'bilinear', 'bicubic'])
                    final_resized = F.interpolate(final_resized, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                    final_resized = filter2D(final_resized, self.sinc_kernel)
                    jpeg_p = final_resized.new_zeros(final_resized.size(0)).uniform_(*self.opt['jpeg_range2'])
                    final_resized = torch.clamp(final_resized, 0, 1)
                    final_resized = self.jpeger(final_resized, quality=jpeg_p)
                else:
                    jpeg_p = final_resized.new_zeros(final_resized.size(0)).uniform_(*self.opt['jpeg_range2'])
                    final_resized = torch.clamp(final_resized, 0, 1)
                    final_resized = self.jpeger(final_resized, quality=jpeg_p)
                    mode = random.choice(['area', 'bilinear', 'bicubic'])
                    final_resized = F.interpolate(final_resized, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                    final_resized = filter2D(final_resized, self.sinc_kernel)

            # 后处理
            self.lq = torch.clamp((final_resized * 255.0).round(), 0, 255) / 255.
            gt_size = self.opt['gt_size']
            (self.gt, self.gt_usm), self.lq = paired_random_crop([self.gt, self.gt_usm], self.lq, gt_size, self.opt['scale'])

            # 更新队列
            self._dequeue_and_enqueue()
            self.gt_usm = self.usm_sharpener(self.gt)
            self.lq = self.lq.contiguous()

        else:
            # 非训练模式
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)
                self.gt_usm = self.usm_sharpener(self.gt)

    def apply_distortion(self, image, distortion_strength=0.1):
        """
        应用失真效果。
        :param image: 输入图像 (Tensor)
        :param distortion_strength: 失真强度
        :return: 失真后的图像
        """
        batch_size, _, height, width = image.size()
        # 创建网格
        x_coords = torch.linspace(-1, 1, width).view(1, 1, 1, -1).expand(batch_size, -1, height, -1)
        y_coords = torch.linspace(-1, 1, height).view(1, 1, -1, 1).expand(batch_size, -1, -1, width)
        grid = torch.cat((x_coords, y_coords), dim=1).to(image.device)

        # 添加随机扰动
        distortion = torch.randn_like(grid) * distortion_strength
        distorted_grid = grid + distortion

        # 归一化到 [-1, 1] 范围
        distorted_grid = torch.clamp(distorted_grid.permute(0, 2, 3, 1), -1, 1)

        # 使用 grid_sample 进行采样
        distorted_image = F.grid_sample(image, distorted_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return distorted_image

    def adjust_brightness(self, image, brightness_factor=None):
        """
        调整图像亮度。
        :param image: 输入图像 (Tensor)
        :param brightness_factor: 亮度因子，默认为随机值 [0.8, 1.2]
        :return: 调整亮度后的图像
        """
        if brightness_factor is None:
            brightness_factor = np.random.uniform(0.8, 1.2)
        return torch.clamp(image * brightness_factor, 0, 1)

    def rotate_image(self, image, angle=None):
        """
        旋转图像。
        :param image: 输入图像 (Tensor)
        :param angle: 旋转角度，默认为随机值 [-15, 15]
        :return: 旋转后的图像
        """
        if angle is None:
            angle = np.random.uniform(-15, 15)
        angle_rad = torch.tensor(angle * np.pi / 180.0)

        batch_size, _, height, width = image.size()
        # 创建旋转矩阵
        rot_matrix = torch.tensor([
            [torch.cos(angle_rad), -torch.sin(angle_rad), 0],
            [torch.sin(angle_rad), torch.cos(angle_rad), 0]
        ]).unsqueeze(0).repeat(batch_size, 1, 1).to(image.device)

        # 创建网格
        grid = F.affine_grid(rot_matrix[:, :2], image.size(), align_corners=True)
        rotated_image = F.grid_sample(image, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return rotated_image

    def downsample_image(self, image, scale_factor=None):
        """
        下采样图像。
        :param image: 输入图像 (Tensor)
        :param scale_factor: 缩放因子，默认为随机值 [0.5, 1.0]
        :return: 下采样后的图像
        """
        if scale_factor is None:
            scale_factor = np.random.uniform(0.5, 1.0)
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        return F.interpolate(image, scale_factor=scale_factor, mode=mode)

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
        super(AMHDMSRGANModel, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True

    def optimize_parameters(self, current_iter):
        # usm sharpening
        l1_gt = self.gt_usm
        percep_gt = self.gt_usm
        gan_gt = self.gt_usm
        if self.opt['l1_gt_usm'] is False:
            l1_gt = self.gt
        if self.opt['percep_gt_usm'] is False:
            percep_gt = self.gt
        if self.opt['gan_gt_usm'] is False:
            gan_gt = self.gt

        # 优化 net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, l1_gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, percep_gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
            # gan loss
            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            l_g_total.backward()
            self.optimizer_g.step()

        # 优化 net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # real
        real_d_pred = self.net_d(gan_gt)
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        l_d_real.backward()
        # fake
        fake_d_pred = self.net_d(self.output.detach().clone())  # clone for pt1.9
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        l_d_fake.backward()
        self.optimizer_d.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)