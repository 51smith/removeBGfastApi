import os
import sys
import copy
import warnings

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image



class Deeplab3BGRemove() :
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    model.eval()

    def __init__(self):
        pass

    def make_transparent_foreground(self, pic, mask):
        # split the image into channels
        b, g, r = cv2.split(np.array(pic).astype('uint8'))
        # add an alpha channel with and fill all with transparent pixels (max 255)
        a = np.ones(mask.shape, dtype='uint8') * 255
        # merge the alpha channel back
        alpha_im = cv2.merge([b, g, r, a], 4)
        # create a transparent background
        bg = np.zeros(alpha_im.shape)
        # setup the new mask
        new_mask = np.stack([mask, mask, mask, mask], axis=2)
        # copy only the foreground color pixels from the original image where mask is set
        foreground = np.where(new_mask, alpha_im, bg).astype(np.uint8)

        return foreground

    def remove_background(self, model, input_file):
        input_image = Image.open(input_file)
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')

        with torch.no_grad():
            output = model(input_batch)['out'][0]
        output_predictions = output.argmax(0)

        # create a binary (black and white) mask of the profile foreground
        mask = output_predictions.byte().cpu().numpy()
        background = np.zeros(mask.shape)
        bin_mask = np.where(mask, 255, background).astype(np.uint8)

        foreground = self.make_transparent_foreground(input_image, bin_mask)

        return foreground, bin_mask

    def custom_background(background_file, foreground):
        final_foreground = Image.fromarray(foreground)
        background = Image.open(background_file)
        x = (background.size[0] - final_foreground.size[0]) / 2 + 0.5
        y = (background.size[1] - final_foreground.size[1]) / 2 + 0.5
        box = (x, y, final_foreground.size[0] + x, final_foreground.size[1] + y)
        crop = background.crop(box)
        final_image = crop.copy()
        # put the foreground in the centre of the background
        paste_box = (0, final_image.size[1] - final_foreground.size[1], final_image.size[0], final_image.size[1])
        final_image.paste(final_foreground, paste_box, mask=final_foreground)
        return final_image

    def image(self, image, output_file):
        foreground, bin_mask = self.remove_background(self.model, image)

        img_fg = Image.fromarray(foreground)
        if foreground_file.endswith(('jpg', 'jpeg')):
            img_fg = img_fg.convert('RGB')

        img_fg.save(foreground_file)


        final_image = self.custom_background(image, foreground)
        final_image.save(output_file)


    def folder(self, foldername, background=False, output='output/'):
        output = self.dir_check(output)
        foldername = self.dir_check(foldername)

        for filename in os.listdir(foldername):
            try:
                self.im_name = filename
                im = self.file_load(foldername+filename)
                im = self.pre_process(im)
                _, _, matte = MODNetBGRemove.modnet(im, inference=False)
                matte = self.post_process(matte, background)
                status = self.save(matte, output, background)
                print(status)
            except:
                print('There is an error for {} file/folder'.format(foldername+filename))

    def save(self, matte, output_path='output/', background=False):
        name = '.'.join(self.im_name.split('.')[:-1])+'.png'
        path = os.path.join(output_path, name)

        if background:
            try:
                matte = cv2.cvtColor(matte, cv2.COLOR_RGB2BGR)
                cv2.imwrite(path, matte)
                return "Successfully saved {}".format(path), name
            except:
                return "Error while saving {}".format(path), ''
        else:
            w, h, _ = matte.shape
            png_image = np.zeros((w, h, 4))
            png_image[:, :, :3] = matte
            png_image[:, :, 3] = self.alpha
            png_image = png_image.astype(np.uint8)
            try:
                png_image = cv2.cvtColor(png_image, cv2.COLOR_RGBA2BGRA)
                cv2.imwrite(path, png_image, [
                            int(cv2.IMWRITE_PNG_COMPRESSION), 9])
                return "Successfully saved {}".format(path), name
            except:
                return "Error while saving {}".format(path), ''
