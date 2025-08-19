import argparse
import json
import os
import random
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 10],
                        help="Range of seeds")
    args = parser.parse_args()
    return args


def generate_seeds(args):
    data_path = './trainval_voc2007.json'
    data = json.load(open(data_path))

    new_all_cats = []
    for cat in data['categories']:
        new_all_cats.append(cat)

    id2img = {}
    for i in data['images']:
        id2img[i['id']] = i

    anno = {i: [] for i in ID2CLASS.keys()}
    for a in data['annotations']:
        if a['iscrowd'] == 1:
            continue
        anno[a['category_id']].append(a)

    for i in range(args.seeds[0], args.seeds[1]):
        random.seed(i)
        sample_shots_list = []
        sample_imgs_list = []
        for shots in [5]:
            for c in ID2CLASS.keys():
                img_ids = {}
                for a in anno[c]:
                    if a['image_id'] in img_ids:
                        img_ids[a['image_id']].append(a)
                    else:
                        img_ids[a['image_id']] = [a]

                sample_shots = []
                sample_imgs = []

                while True:
                    imgs = random.sample(list(img_ids.keys()), shots)
                    for img in imgs:
                        skip = False
                        for s in sample_shots:
                            if img == s['image_id']:
                                skip = True
                                break
                        if skip:
                            continue
                        if len(img_ids[img]) + len(sample_shots) > shots:
                            continue
                        sample_shots.extend(img_ids[img])
                        sample_imgs.append(id2img[img])
                        if len(sample_shots) == shots:
                            break
                    if len(sample_shots) == shots:
                        break
                sample_shots_list.append(sample_shots)
                sample_imgs_list.append(sample_imgs)
            all_shots = []
            all_imgs = []
            for k in range(len(sample_shots_list)):
                for v in range(len(sample_shots_list[k])):
                    all_shots.append(sample_shots_list[k][v])
            for k in range(len(sample_imgs_list)):
                for v in range(len(sample_imgs_list[k])):
                    all_imgs.append(sample_imgs_list[k][v])
            new_data = {
                # 'info': data['info'],
                # 'licenses': data['licenses'],
                'images': all_imgs,
                'annotations': all_shots,
            }
            save_path = get_save_path_seeds(data_path, shots, i)
            new_data['categories'] = new_all_cats
            with open(save_path, 'w') as f:
                json.dump(new_data, f)


def get_save_path_seeds(path, shots, seed):
    s = path.split('/')
    prefix = 'full_box_{}shot_all_train'.format(shots)
    save_dir = os.path.join('./VOCdevkit/VOC2007', 'seed' + str(seed))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, prefix + '.json')
    return save_path


if __name__ == '__main__':
    ID2CLASS = {
        1: "aeroplane",
        2: "bicycle",
        3: "bird",
        4: "boat",
        5: "bottle",
        6: "bus",
        7: "car",
        8: "cat",
        9: "chair",
        10: "cow",
        11: "diningtable",
        12: "dog",
        13: "horse",
        14: "motorbike",
        15: "person",
        16: "pottedplant",
        17: "sheep",
        18: "sofa",
        19: "train",
        20: "tv"
    }

    CLASS2ID = {v: k for k, v in ID2CLASS.items()}

    args = parse_args()
    generate_seeds(args)
