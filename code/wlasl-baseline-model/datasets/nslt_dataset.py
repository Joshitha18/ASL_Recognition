import json
import math
import os
import os.path
import random

import cv2
import numpy as np
import torch
import torch.utils.data as data_utl


def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))


def load_rgb_frames(image_dir, vid, start, num):
    frames = []
    for i in range(start, start + num):
        try:
            img = cv2.imread(os.path.join(image_dir, vid, "image_" + str(i).zfill(5) + '.jpg'))[:, :, [2, 1, 0]]
        except:
            print(os.path.join(image_dir, vid, str(i).zfill(6) + '.jpg'))
        w, h, c = img.shape
        if w < 226 or h < 226:
            d = 226. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
        img = (img / 255.) * 2 - 1
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)


def load_rgb_frames_from_video(vid_root, vid, start, num, resize=(256, 256)):
    video_path = os.path.join(vid_root, vid + '.mp4')

    vidcap = cv2.VideoCapture(video_path)

    frames = []

    total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for offset in range(min(num, int(total_frames - start))):
        success, img = vidcap.read()

        ## Add by Ray - get out if we have no more frames
        if success == False:
            break

        w, h, c = img.shape
        if w < 226 or h < 226:
            d = 226. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)

        if w > 256 or h > 256:
            img = cv2.resize(img, (math.ceil(w * (256 / w)), math.ceil(h * (256 / h))))

        img = (img / 255.) * 2 - 1

        frames.append(img)

    return np.asarray(frames, dtype=np.float32)


def load_flow_frames(image_dir, vid, start, num):
    frames = []
    for i in range(start, start + num):
        imgx = cv2.imread(os.path.join(image_dir, vid, vid + '-' + str(i).zfill(6) + 'x.jpg'), cv2.IMREAD_GRAYSCALE)
        imgy = cv2.imread(os.path.join(image_dir, vid, vid + '-' + str(i).zfill(6) + 'y.jpg'), cv2.IMREAD_GRAYSCALE)

        w, h = imgx.shape
        if w < 224 or h < 224:
            d = 224. - min(w, h)
            sc = 1 + d / min(w, h)
            imgx = cv2.resize(imgx, dsize=(0, 0), fx=sc, fy=sc)
            imgy = cv2.resize(imgy, dsize=(0, 0), fx=sc, fy=sc)

        imgx = (imgx / 255.) * 2 - 1
        imgy = (imgy / 255.) * 2 - 1
        img = np.asarray([imgx, imgy]).transpose([1, 2, 0])
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)


def make_dataset(split_file, split, root, mode, num_classes):
    dataset = []
    ## Open split file JSON
    with open(split_file, 'r') as f:
        data = json.load(f)

    i = 0
    count_skipping = 0
    
    ## Iterate through each video ID
    for vid in data.keys():
        ## Only look for the videos that are given the train label in the split
        ## file JSON
        i += 1
        vid_root = root['word']

        ## Get directory where the videos are
        video_path = os.path.join(vid_root, vid + '.mp4')
        print('Video #{} / {} - {}'.format(i, len(data), video_path))
        if split == 'train':
            if data[vid]['subset'] not in ['train', 'val']:
                print('{} is not in the train or test set - skipping'.format(video_path))
                count_skipping += 1
                continue
        else:
            if data[vid]['subset'] != 'test':
                print('{} is not in the test set - skipping'.format(video_path))
                count_skipping += 1
                continue
                
        ## Get full path to video
        # If the video doesn't exist, skip
        if not os.path.exists(video_path):
            count_skipping += 1
            print('{} does not exist - skipping'.format(video_path))
            continue
        
        ## Get number of frames for the video
        num_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
        
        ## If we are dealing with optical flow, we only need half of the num_frames
        ## as each "iteration" will have two frames
        if mode == 'flow':
            num_frames = num_frames // 2
        
        ## If there are less than 10 frames, there isn't enough information to
        ## decipher what sign it is - skip this too
        if num_frames - 0 < 9:
            print("Skip video ", vid)
            count_skipping += 1
            continue
        
        # Get label
        label = int(data[vid]['action'][0])
        
        if len(vid) == 5:
            ## We have tuples where we know the ID of the video,
            ## the label for each frame... not sure what src is doing but
            ## we can ignore
            ## 4th element is the frame where the action starts
            ## 5th element is how many frames this action lasts
            dataset.append((vid, label, 0, data[vid]['action'][2] - data[vid]['action'][1]))
        elif len(vid) == 6:  ## sign kws instances ---> This doesn't run in the training set
            dataset.append((vid, label, 0, data[vid]['action'][1], data[vid]['action'][2] - data[vid]['action'][1]))

    print("Skipped videos: ", count_skipping)
    print("Total number of videos in the dataset: ", len(dataset))
    return dataset


def get_num_class(split_file):
    classes = set()

    content = json.load(open(split_file))

    for vid in content.keys():
        class_id = content[vid]['action'][0]
        classes.add(class_id)

    return len(classes)


class NSLT(data_utl.Dataset):

    def __init__(self, split_file, split, root, mode, transforms=None):
        self.num_classes = get_num_class(split_file)

        self.data = make_dataset(split_file, split, root, mode, num_classes=self.num_classes)
        self.split_file = split_file
        self.transforms = transforms
        self.mode = mode
        self.root = root

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        vid, label, start_frame, nf = self.data[index]

        total_frames = 64
        
        ## Get the start frame within the valid range
        ## Choose any one of them
        try:
            start_f = random.randint(0, nf - total_frames - 1) + start_frame
        except ValueError:
            start_f = start_frame

        # From this start frame, grab 64 consecutive frames
        # and make sure we normalize the frames from [-1, 1]
        # Output should be 64 x 256 x 256 x 3
        imgs = load_rgb_frames_from_video(self.root['word'], vid, start_f, total_frames)

        # If we can't load this video for some reason, signal a skip
        if imgs.shape[0] == 0:
            imgs = np.zeros((total_frames, 224, 224, 3), dtype=np.float32)
            print(os.path.join(self.root['word'], vid + '.mp4') + ' could not be read for some reason.  Skipping')
            label = -1
        else:
            # If we don't end up having 64 frames, then we pad the video sequence
            # to get up to 64
            # We randomly choose the first or last frame and tack it onto the end
            imgs = self.pad(imgs, total_frames)
            
            # Run through the data augmentation
            # 64 x 224 x 224 x 3
            imgs = self.transforms(imgs)
        
        # Convert frames and labels to torch tensor
        # Return a 1D tensor of size total_frames that contains the label
        labels = label * np.ones(total_frames, dtype=np.int)
        ret_lab = torch.from_numpy(labels)
        
        # frames - 3 x 64 x 224 x 224
        ret_img = video_to_tensor(imgs)

        return ret_img, ret_lab, vid

    def __len__(self):
        return len(self.data)

    def pad(self, imgs, total_frames):
        if imgs.shape[0] < total_frames:
            num_padding = total_frames - imgs.shape[0]

            if num_padding:
                prob = np.random.random_sample()
                if prob > 0.5:
                    pad_img = imgs[0]
                    pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
                    padded_imgs = np.concatenate([imgs, pad], axis=0)
                else:
                    pad_img = imgs[-1]
                    pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
                    padded_imgs = np.concatenate([imgs, pad], axis=0)
        else:
            padded_imgs = imgs
        
        return padded_imgs

    @staticmethod
    def pad_wrap(imgs, total_frames):
        if imgs.shape[0] < total_frames:
            num_padding = total_frames - imgs.shape[0]

            if num_padding:
                pad = imgs[:min(num_padding, imgs.shape[0])]
                k = num_padding // imgs.shape[0]
                tail = num_padding % imgs.shape[0]

                pad2 = imgs[:tail]
                if k > 0:
                    pad1 = np.array(k * [pad])[0]

                    padded_imgs = np.concatenate([imgs, pad1, pad2], axis=0)
                else:
                    padded_imgs = np.concatenate([imgs, pad2], axis=0)
        else:
            padded_imgs = imgs

        #label = label[:, 0]
        #label = np.tile(label, (total_frames, 1)).transpose((1, 0))

        return padded_imgs#, label

