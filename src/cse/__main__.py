import torch
from ml_utils.models import gradcam_model
import torchvision.transforms as transforms
from PIL import Image
from cse.cse_algorithms import *
from pathlib import Path
import glob

import warnings
warnings.filterwarnings("ignore")


LABELS_MAP_FASHION = {
    0: '0',
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6',
    7: '7',
    8: '8',
    9: '9'
}


ATTR_MAP = {
    'grad_cam': 0,
    'grad_cam++': 1,
    'full_grad': 2,
    'x_grad_cam': 3,
    'ablation_cam': 4
}


SEG_MAP = {
    'slic': 0,
    'bass': 1,
    'felzen': 2,
    'watershed': 3
}


def main_old(attr_map: int = 0,
             seg_map: int = 0,
             output_class: int = 0,
             img_dir: str = 'data/train'):

    class_target = output_class
    top_n_start = 1
    top_n_stop = 20
    threshold = 0.90
    pruning_heuristic = 1
    # batch size of 16 appears to be optimal for search speed
    # can do 20 regions in ~3 minutes!
    batch_sz = 16
    model_dict = torch.load(
        '/workspace/adv_robustness/region_explainability/mnist_training/resnet_models/grad_cam_model.pt')
    model = gradcam_model()
    model.load_state_dict(model_dict)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device)
    model.eval()
    # torch.jit.trace speeds up evaluation

    # good_img_transform = transforms.Normalize((0.1307,), (0.3081,))
    # This is to reverse the normalization done to the images that centered them around imagenet mean and std
    # The invTrans should be used on images before saving them.
    # invTrans = transforms.Normalize((1/0.1307,), (1/0.3081,))

    images = Image.open(img_dir)
    img_dir, img_name = (img_dir.split('/')[:-2], img_dir.split('/')[-1])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    inv_img = transform(images).unsqueeze(0)
    img_np = inv_img.detach().cpu().squeeze().numpy()
    # plt.imshow(img_np)
    # compactness=50

    input_tensor = transform(images).unsqueeze(0).to(device)
    targets = [ClassifierOutputTarget(class_target)]
    target_layers = [model.layer2]

    cam = None
    if attr_map == 0:
        cam = GradCAM(model=model, target_layers=target_layers)

    if attr_map == 1:
        cam = GradCAMPlusPlus(model=model, target_layers=target_layers)

    if attr_map == 2:
        cam = FullGrad(model=model, target_layers=target_layers)

    if attr_map == 3:
        cam = XGradCAM(model=model, target_layers=target_layers)

    if attr_map == 4:
        cam = AblationCAM(model=model, target_layers=target_layers)

    segments = None
    if seg_map == 0:
        segments = slic(img_np, n_segments=25, compactness=1, start_label=1)

    if seg_map == 2:
        segments = felzenszwalb(img_np, scale=5, sigma=0.5, min_size=5)

    if seg_map == 3:
        segments = watershed(img_np, markers=25, compactness=0.001)

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    # fast_model = torch.jit.trace(model, torch.zeros(batch_sz, 1, 28, 28).to(device))

    working_example = region_explainability(image=input_tensor, segment_mask=segments, top_n_start=top_n_start,
                                            model=model, SMU_class_index=class_target,
                                            threshold=threshold, top_n_stop=top_n_stop,
                                            MAX_BATCH_SZ=batch_sz,
                                            PRUNE_HEURISTIC=pruning_heuristic)

    torch.save(working_example, 'labelme/MNIST_71/metric_results/' + img_name)
    if working_example == -1:
        return -1


def main(attr_map: str,
         seg_map: str = 0,
         class_index: int = 0,
         img_dir: str = 'data/train'):

    # Psuedo code for main func:

    # build model & jit model
    # get images folder
    images = list()

    model = None
    jit_model = model

    for image in images:
        output = jit_model(image)
        output_class_index = output

        if output_class_index != class_index:
            print(f'Warning, predicted class index is not the same inputted class index')
            print('Skipping over image.')
            continue

        # seg_map = get_seg_map(image, seg_type='slic')

        # attr_map = get_attr_map(model, class_index, image)

        # avg_score_attr_map = get_avg_score_attr_map(seg_map, attr_map)

        # ranked_attr_map = get_attr_rank(avg_score_attr_map)

        # features = get_feature_masks(image=image, attributions=ranked_attr_map,
        #                            seg_map=seg_map)

        # counterfactual_search_result = cse(image, jit_model, features, start_depth, end_depth)

    pass


def test():
    pass


if __name__ == "__main__":
    import argparse
    import glob

    parser = argparse.ArgumentParser(description='region explainability')
    parser.add_argument('--attr_map', default='grad_cam', type=str,
                        help=("attribution method to use, it can be:\n 'gradcam'," +
                              " 'gradcam++', 'fullgrad', 'x_grad_cam', or 'ablation_cam'"))

    parser.add_argument('--seg_map', default='slic', type=str, help=(
                        "segmentation method to use, it can be: 'slic', 'bass', 'felzen', or 'watershed'"))

    parser.add_argument('--class_index', default=7, type=int,
                        help='class index of the source class')

    parser.add_argument('--image_dir', default='./',
                        type=str, help='directory path of images for cse algorithm')

    args = parser.parse_args()

    # main_old(attr_map=args.attr_map,
    #          seg_map=args.seg_map,
    #          output_class=args.class_index,
    #          img_dir=args.image_dir)

    

    print('Completed.')
