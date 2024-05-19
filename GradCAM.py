# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision import models
import matplotlib.pyplot as plt
from models import LHKDNET
from models.LMROD import get_model_name
import shutil

class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(
                    self.save_activation))
            # Backward compatibility with older pytorch versions:
            if hasattr(target_layer, 'register_full_backward_hook'):
                self.handles.append(
                    target_layer.register_full_backward_hook(
                        self.save_gradient))
            else:
                self.handles.append(
                    target_layer.register_backward_hook(
                        self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu().detach()] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()

heatpath = "C:/YyFiles/tar_00390.jpg"
class GradCAM:
    def __init__(self, model, target_layers, reshape_transform=None):
        self.model = model.eval()
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    @staticmethod
    def get_cam_weights(grads):
        return np.mean(grads, axis=(2, 3), keepdims=True)

    @staticmethod
    def get_loss(output, target_category):
        loss = 0
        for i in range(len(target_category)):
            loss = loss + output[i, target_category[i]]
        return loss

    def get_cam_image(self, activations, grads):
        weights = self.get_cam_weights(grads)
        weighted_activations = weights * activations
        cam = weighted_activations.sum(axis=1)

        return cam

    @staticmethod
    def get_target_width_height(input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def compute_cam_per_layer(self, input_tensor):
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer

        for layer_activations, layer_grads in zip(activations_list, grads_list):
            cam = self.get_cam_image(layer_activations, layer_grads)
            cam[cam < 0] = 0  # works like mute the min-max scale in the function of scale_cam_image
            scaled = self.scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer):
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result)

    @staticmethod
    def scale_cam_image(cam, target_size=None):
        result = []
        for img in cam:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)

        return result

    def __call__(self, input_tensor, target_category=None):

        # 正向传播得到网络输出logits(未经过softmax)
        output = self.activations_and_grads(input_tensor)
        if isinstance(target_category, int):
            target_category = [target_category] * input_tensor.size(0)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
            print(f"category id: {target_category}")
        else:
            assert (len(target_category) == input_tensor.size(0))

        self.model.zero_grad()
        loss = self.get_loss(output, target_category)
        loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor)
        return self.aggregate_multi_layers(cam_per_layer)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.
    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def image_proprecess(img_path):
    img = Image.open(img_path)
    data_transforms = transforms.Compose([
        transforms.Resize((384, 384), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    data = data_transforms(img)
    data = torch.unsqueeze(data, 0)
    img_resize = img.resize((384, 384))
    return img_resize, data


def Init_Setting_00280(img_name: str):
    model = None
    if (img_name == "LHKDNET"):
        model = models.squeezenet1_1(pretrained=True)
    elif img_name == "LMROD":
        model = models.squeezenet1_0(pretrained=True)
    else:
        model = models.resnet152(pretrained=True)
    model = model.cuda().eval()
    return model

    #model = models.resnet152(pretrained=True)
    # model = models.mobilenet_v3_large(pretrained=True)
    #model = models.mobilenet_v2(pretrained=True)
    # model = models.mobilenet_v3_small(pretrained=True)
    #model = models.vgg19(pretrained=True)
    # model = models.squeezenet1_0(pretrained=True)
    # model = models.squeezenet1_1(pretrained=True)
    # model = models.shufflenet_v2_x1_0(pretrained=True)


def Init_Setting_00293(img_name: str):
    model = None
    if (img_name == "LHKDNET"):
        model = models.vgg19(pretrained=True)
    elif img_name == "LMROD":
        model = models.squeezenet1_1(pretrained=True)
    else:
        model = models.resnet152(pretrained=True)
    model = model.cuda().eval()
    return model


def Init_Setting_00390(img_name: str):
    model = None
    if img_name == "LHKDNET":
        model = models.resnet152(pretrained=True)
    elif img_name == "LMROD":
        model = models.shufflenet_v2_x2_0()
    else:
        model = models.resnet152(pretrained=True)
    model = model.cuda().eval()
    return model


def Init_Setting_LMROD():
    # model = models.resnet152(pretrained=True)
    # model = models.mobilenet_v3_large(pretrained=True)
    #model = models.mobilenet_v2(pretrained=True)
    # model = models.mobilenet_v3_small(pretrained=True)
    model = models.vgg19(pretrained=True)
    #model = models.squeezenet1_0(pretrained=True)
    # model = models.squeezenet1_1(pretrained=True)
    # model = models.shufflenet_v2_x2_0(pretrained=True)
    model = model.cuda().eval()
    return model


#00280: LMROD: squ_1_0_features_12 ; LHKDNET:squ_1_1_features_12

if __name__ == "__main__":
    rootpath = "C:/YyFiles/papercode/LMROD-Lite/img_heatmaps/target/"
    image_name = "00293.jpg"
    imgs_path = rootpath + image_name
    savepath = "C:/YyFiles/papercode/LMROD-Lite/img_heatmaps/savefile/"
    if image_name == "00280.jpg":
        model_LHKDNET = Init_Setting_00280("LHKDNET")
        model_LMROD = Init_Setting_00280("LMROD")
        target_layers_LHKDNET = [model_LHKDNET.features[12]]
        img, data = image_proprecess(imgs_path)

        cam = GradCAM(model=model_LHKDNET, target_layers=target_layers_LHKDNET)
        target_category = None

        data = data.cuda()
        grayscale_cam = cam(input_tensor=data, target_category=target_category)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(np.array(img) / 255.,
                                          grayscale_cam,
                                          use_rgb=True)
        plt.show()
        plt.imshow(visualization)
        plt.xticks()
        plt.yticks()
        plt.axis('off')
        plt.savefig("C:/YyFiles/papercode/LMROD-Lite/img_heatmaps/savefile/00280_heatmap_LHKDNET.jpg")

        target_layers_LMROD = [model_LMROD.features[12]]
        img, data = image_proprecess(imgs_path)

        cam = GradCAM(model=model_LMROD, target_layers=target_layers_LMROD)
        target_category = None

        data = data.cuda()
        grayscale_cam = cam(input_tensor=data, target_category=target_category)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(np.array(img) / 255.,
                                          grayscale_cam,
                                          use_rgb=True)
        plt.show()
        plt.imshow(visualization)
        plt.xticks()
        plt.yticks()
        plt.axis('off')
        plt.savefig("C:/YyFiles/papercode/LMROD-Lite/img_heatmaps/savefile/00280_heatmap_LMROD.jpg")

    elif image_name == "00293.jpg":
        model_LHKDNET = Init_Setting_00293("LHKDNET")
        model_LMROD = Init_Setting_00293("LMROD")

        target_layers_LHKDNET = [model_LHKDNET.features[36]]

        img, data = image_proprecess(imgs_path)
        cam = GradCAM(model=model_LHKDNET, target_layers=target_layers_LHKDNET)
        target_category = None
        data = data.cuda()
        grayscale_cam = cam(input_tensor=data, target_category=target_category)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(np.array(img) / 255.,
                                          grayscale_cam,
                                          use_rgb=True)
        plt.show()
        plt.imshow(visualization)
        plt.xticks()
        plt.yticks()
        plt.axis('off')
        plt.savefig("C:/YyFiles/papercode/LMROD-Lite/img_heatmaps/savefile/00293_heatmap_LHKDNET.jpg")

        target_layers_LMROD = [model_LMROD.features[12]]
        img, data = image_proprecess(imgs_path)
        cam = GradCAM(model=model_LMROD, target_layers=target_layers_LMROD)
        target_category = None
        data = data.cuda()
        grayscale_cam = cam(input_tensor=data, target_category=target_category)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(np.array(img) / 255.,
                                          grayscale_cam,
                                          use_rgb=True)
        plt.show()
        plt.imshow(visualization)
        plt.xticks()
        plt.yticks()
        plt.axis('off')
        plt.savefig("C:/YyFiles/papercode/LMROD-Lite/img_heatmaps/savefile/00293_heatmap_LMROD.jpg")

    elif image_name == "00390.jpg":
        model_LHKDNET = Init_Setting_00390("LHKDNET")

        model_LMROD = Init_Setting_00390("LMROD")

        target_layers_LHKDNET = [model_LHKDNET.layer4[1]]

        img, data = image_proprecess(imgs_path)
        cam = GradCAM(model=model_LHKDNET, target_layers=target_layers_LHKDNET)
        target_category = None
        data = data.cuda()
        grayscale_cam = cam(input_tensor=data, target_category=target_category)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(np.array(img) / 255.,
                                          grayscale_cam,
                                          use_rgb=True)
        plt.show()
        plt.imshow(visualization)
        plt.xticks()
        plt.yticks()
        plt.axis('off')
        plt.savefig("C:/YyFiles/papercode/LMROD-Lite/img_heatmaps/savefile/00390_heatmap_LHKDNET.jpg")

        target_layers_LMROD = [model_LMROD.stage4[-2]]
        img, data = image_proprecess(imgs_path)
        cam = GradCAM(model=model_LMROD, target_layers=target_layers_LMROD)
        target_category = None
        data = data.cuda()
        grayscale_cam = cam(input_tensor=data, target_category=target_category)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(np.array(img) / 255.,
                                          grayscale_cam,
                                          use_rgb=True)
        plt.show()
        plt.imshow(visualization)
        plt.xticks()
        plt.yticks()
        plt.axis('off')
        target_path = "C:/YyFiles/papercode/LMROD-Lite/img_heatmaps/savefile/00390_heatmap_LMROD.jpg"
        plt.savefig(target_path)
        shutil.copy(heatpath,target_path)
    # imgs_path = "C:/YyFiles/papercode/LMROD-Lite/img_heatmaps/target/00280.jpg"
    # # target_layers = [model.features[12]]
    # # target_layers = [model.stage3[2]]
    # # target_layers = [model.layer2[-2]]
    # img, data = image_proprecess(imgs_path)
    #
    # cam = GradCAM(model=model, target_layers=target_layers)
    # target_category = None
    #
    # data = data.cuda()
    # grayscale_cam = cam(input_tensor=data, target_category=target_category)
    # grayscale_cam = grayscale_cam[0, :]
    # visualization = show_cam_on_image(np.array(img) / 255.,
    #                                   grayscale_cam,
    #                                   use_rgb=True)
    # plt.show()
    # plt.imshow(visualization)
    # plt.xticks()
    # plt.yticks()
    # plt.axis('off')
    # plt.savefig("C:/YyFiles/papercode/LMROD-Lite/img_heatmaps/savefile/00280_heatmap_LHKDNET.jpg")
    # print("热力图生成")
