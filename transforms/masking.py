import numpy as np
import math
import random
from einops import rearrange

class TubeMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame =  self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame 
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        mask_per_frame = np.hstack([
            np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
            np.ones(self.num_masks_per_frame),
        ])
        np.random.shuffle(mask_per_frame)
        mask = np.tile(mask_per_frame, (self.frames,1)).flatten()
        return mask 


class CubeMaskingGenerator:
    def __init__(
            self, input_size=(8,14,14), mask_ratio=0.4, min_num_patches=16, max_num_patches=None,
            min_aspect=0.3, max_aspect=None, type_3d='cube'):
        self.temporal ,self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = int(self.num_patches * mask_ratio)
        self.num_masking_frames = self.temporal

        self.min_num_patches = min_num_patches # smaller than max_num_patches
        self.max_num_patches = self.num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.type_3d = type_3d
        
    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height, self.width, self.min_num_patches, self.max_num_patches,
            self.num_masking_patches, self.log_aspect_ratio[0], self.log_aspect_ratio[1])
        return repr_str

    def get_shape(self):
        return self.temporal, self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for attempt in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top: top + h, left: left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self):
        time_marker = np.zeros(shape=self.temporal, dtype=np.int32)
        cube_mask = np.zeros(shape=self.get_shape(), dtype=np.int32)
        cube_marker = []
        temp_mask_count = 0
        while temp_mask_count < self.num_masking_frames:
            # generate 2D block-wise mask
            mask = np.zeros(shape=self.get_shape()[1:], dtype=np.int32)
            mask_count = 0
            while mask_count < self.num_masking_patches:
                max_mask_patches = self.num_masking_patches - mask_count
                max_mask_patches = min(max_mask_patches, self.max_num_patches)

                delta = self._mask(mask, max_mask_patches)
                if delta == 0:
                    break
                else:
                    mask_count += delta

            rem_mask_patches = self.num_masking_patches - mask.sum()
            if rem_mask_patches > 0:
                idx_to_choice = np.where(mask==0)
                idx = np.random.choice(np.arange(len(idx_to_choice[0])), rem_mask_patches, replace=False)
                mask[idx_to_choice[0][idx],idx_to_choice[1][idx]]=1
            elif rem_mask_patches < 0:
                idx_to_choice = np.where(mask==1)
                idx = np.random.choice(np.arange(len(idx_to_choice[0])), -rem_mask_patches, replace=False)
                mask[idx_to_choice[0][idx],idx_to_choice[1][idx]]=0

            # assign to cube mask
            if self.type_3d == 'cube':
                start_frame = random.randint(0, self.temporal)
                accumulate_frames = random.randint(1, self.num_masking_frames - temp_mask_count)
            elif self.type_3d == 'tube':
                start_frame = 0
                accumulate_frames = self.temporal
            mask_count = 0
            for i in range(start_frame, start_frame+accumulate_frames):
                if i > self.temporal-1:
                    break
                if time_marker[i] == 0: # only update the unmask frame
                    time_marker[i] = 1
                    cube_mask[i] = mask
                    mask_count+=1
                else: #avoid to overlap the orginal mask
                    break
            temp_mask_count += mask_count
            if mask_count > 0: # mark the center frame index(mask_count > 0)
                cube_marker.append([start_frame, mask_count])
    
        return rearrange(cube_mask, 't h w -> (t h w)')

        # return np.stack(masks).flatten() #, cube_marker