# Our libs
from networks.transforms import trimap_transform, groupnorm_normalise_image
from networks.models import build_model
from dataloader import PredDataset

# System libs
import os
import argparse

# External libs
import cv2
import numpy as np
import torch
import json


def gen_trimap(alpha, ksize=3, iterations=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = cv2.dilate(alpha, kernel, iterations=iterations)
    eroded = cv2.erode(alpha, kernel, iterations=iterations)
    trimap = np.zeros(alpha.shape) + 128
    trimap[eroded >= 255] = 255
    trimap[dilated <= 0] = 0
    return trimap


def load_samples(fg_dir, bg_dir, filelist):
    names = ['defocus', 'fire', 'fur', 'glass_ice', 'hair_easy', 'hair_hard', 
             'insect', 'logo', 'motion', 'net', 'plant_flower', 'plant_leaf', 'plant_tree', 
             'plastic_bag', 'sharp', 'smoke_cloud_easy', 'spider_web', 'text', 'texture_holed', 
             'texture_smooth', 'water_drop', 'water_spray']
    name2class = {name:idx for idx, name in enumerate(names)}

    print(names)

    fg_dir = os.path.join(fg_dir, 'test')
    bg_dir = os.path.join(bg_dir, 'VOCdevkit/VOC2012/JPEGImages')

    samples = []
    for line in open(filelist, encoding='utf-8').read().splitlines():
        name, fg_name, bg_name = line.split(':')
        if name not in name2class or name not in names: continue
        alpha_path = os.path.join(fg_dir, name, 'alpha', fg_name)
        if not os.path.exists(alpha_path): continue
        fg_path = os.path.join(fg_dir, name, 'fg', fg_name)
        if not os.path.exists(fg_path): continue
        mask_path = os.path.join(fg_dir, name, 'mask', fg_name.replace('.jpg', '.json'))
        if not os.path.exists(mask_path): continue
        bg_path = os.path.join(bg_dir, bg_name)
        if not os.path.exists(bg_path): continue
        masks = json.load(open(mask_path))['shapes']
        regions = []
        for mask in masks:
            if mask['label']=='keep':
                (x1, y1), (x2, y2) = mask['points']
                if x1>=0 and y1>=0 and x2>x1 and y2>y1:
                    regions.append([[x1, y1], [x2, y2]])
        if len(regions) > 0:
            samples.append((bg_path, alpha_path, fg_path, regions, name, name2class[name]))
    return samples, names, name2class


def composite(bg_path, fg_path, alpha_path, regions):
    fg = cv2.imread(fg_path)
    alpha = cv2.imread(alpha_path)

    r = 1920.0 / max(alpha.shape[:2])
    if r < 1:
        alpha = cv2.resize(alpha, None, fx=r, fy=r)
        fg = cv2.resize(fg, None, fx=r, fy=r)
        regions = np.array(regions) * r
    r = 800.0 / min(alpha.shape[:2])
    if r > 1:
        alpha = cv2.resize(alpha, None, fx=r, fy=r)
        fg = cv2.resize(fg, None, fx=r, fy=r)
        regions = np.array(regions) * r

    h, w ,c = fg.shape
    bg = cv2.imread(bg_path)
    bh, bw, bc = bg.shape

    wratio = float(w) / bw
    hratio = float(h) / bh
    ratio = wratio if wratio > hratio else hratio     
    if ratio > 1:
        new_bw = int(bw * ratio + 1.0)
        new_bh = int(bh * ratio + 1.0)
        bg = cv2.resize(bg, (new_bw, new_bh), interpolation=cv2.INTER_LINEAR)
    bg = bg[0:h, 0:w, :]
    alpha_f = alpha / 255.
    comp = (fg*alpha_f + bg*(1.-alpha_f)).astype(np.uint8)
    return alpha, fg, bg, comp, regions


def np_to_torch(x):
    return torch.from_numpy(x).permute(2, 0, 1)[None, :, :, :].float().cuda()


def scale_input(x: np.ndarray, scale: float, scale_type) -> np.ndarray:
    ''' Scales inputs to multiple of 8. '''
    h, w = x.shape[:2]
    h1 = int(np.ceil(scale * h / 8) * 8)
    w1 = int(np.ceil(scale * w / 8) * 8)
    x_scale = cv2.resize(x, (w1, h1), interpolation=scale_type)
    return x_scale


def predict_fba_folder(model, args):
    save_dir = args.output_dir

    dataset_test = PredDataset(args.image_dir, args.trimap_dir)

    gen = iter(dataset_test)
    for item_dict in gen:
        image_np = item_dict['image']
        trimap_np = item_dict['trimap']

        fg, bg, alpha = pred(image_np, trimap_np, model)

        cv2.imwrite(os.path.join(save_dir, item_dict['name'][:-4] + '_fg.png'), fg[:, :, ::-1] * 255)
        cv2.imwrite(os.path.join(save_dir, item_dict['name'][:-4] + '_bg.png'), bg[:, :, ::-1] * 255)
        cv2.imwrite(os.path.join(save_dir, item_dict['name'][:-4] + '_alpha.png'), alpha * 255)


def predict_sim_folder(model, args):
    samples, names, name2class = load_samples(args.fg_dir, args.bg_dir, args.filelist)

    save_dir = args.output_dir

    dataset_test = PredDataset(args.image_dir, args.trimap_dir)

    sad_list = {}
    for i in range(len(names)):
        sad_list[i] = [0, 0] 
    total_sad = 0
    for i, (bg_path, alpha_path, fg_path, regions, name, idx) in enumerate(samples):
        gt_alpha, fg, bg, comp, regions = composite(bg_path, fg_path, alpha_path, regions)
        trimap = gen_trimap(gt_alpha[:,:,0])
        image_np = cv2.cvtColor(comp, cv2.COLOR_BGR2RGB) / 255.
        trimap_np = np.concatenate([trimap[..., None]==0, trimap[..., None]==255], axis=2).astype(np.float32)
        
        fg, bg, alpha = pred(image_np, trimap_np, model)

        sad = np.sum(np.abs(alpha - gt_alpha[..., 0] / 255.) * (trimap == 128))
        sad_list[idx][0] += sad
        sad_list[idx][1] += 1 
        total_sad += sad
        print("sad(%d) = %f" % (i, sad))

        cv2.imwrite(os.path.join(save_dir, '%04d_fg.png' % i), fg[:, :, ::-1] * 255)
        cv2.imwrite(os.path.join(save_dir, '%04d_bg.png' % i), bg[:, :, ::-1] * 255)
        cv2.imwrite(os.path.join(save_dir, '%04d_alpha.png' % i), alpha * 255)
    for i in range(len(names)):
        print("{} {}".format(names[i], sad_list[i][0]/(sad_list[i][1]+1e-6)))
    print("MeanSAD = %f" % (total_sad / len(samples)))


def pred(image_np: np.ndarray, trimap_np: np.ndarray, model) -> np.ndarray:
    ''' Predict alpha, foreground and background.
        Parameters:
        image_np -- the image in rgb format between 0 and 1. Dimensions: (h, w, 3)
        trimap_np -- two channel trimap, first background then foreground. Dimensions: (h, w, 2)
        Returns:
        fg: foreground image in rgb format between 0 and 1. Dimensions: (h, w, 3)
        bg: background image in rgb format between 0 and 1. Dimensions: (h, w, 3)
        alpha: alpha matte image between 0 and 1. Dimensions: (h, w)
    '''
    h, w = trimap_np.shape[:2]

    image_scale_np = scale_input(image_np, 1.0, cv2.INTER_LANCZOS4)
    trimap_scale_np = scale_input(trimap_np, 1.0, cv2.INTER_LANCZOS4)

    with torch.no_grad():

        image_torch = np_to_torch(image_scale_np)
        trimap_torch = np_to_torch(trimap_scale_np)

        trimap_transformed_torch = np_to_torch(trimap_transform(trimap_scale_np))
        image_transformed_torch = groupnorm_normalise_image(image_torch.clone(), format='nchw')

        output = model(image_torch, trimap_torch, image_transformed_torch, trimap_transformed_torch)

        output = cv2.resize(output[0].cpu().numpy().transpose((1, 2, 0)), (w, h), cv2.INTER_LANCZOS4)
    alpha = output[:, :, 0]
    fg = output[:, :, 1:4]
    bg = output[:, :, 4:7]

    alpha[trimap_np[:, :, 0] == 1] = 0
    alpha[trimap_np[:, :, 1] == 1] = 1
    fg[alpha == 1] = image_np[alpha == 1]
    bg[alpha == 0] = image_np[alpha == 0]
    return fg, bg, alpha


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--encoder', default='resnet50_GN_WS', help="encoder model")
    parser.add_argument('--decoder', default='fba_decoder', help="Decoder model")
    parser.add_argument('--weights', default='FBA.pth')
    parser.add_argument('--image_dir', default='./examples/images', help="")
    parser.add_argument('--trimap_dir', default='./examples/trimaps', help="")
    parser.add_argument('--output_dir', default='./examples/predictions', help="")
    parser.add_argument('--fg_dir', default='', help="")
    parser.add_argument('--bg_dir', default='', help="")
    parser.add_argument('--filelist', default='', help="")

    args = parser.parse_args()
    model = build_model(args)
    model.eval()
    # predict_fba_folder(model, args)
    predict_sim_folder(model, args)
