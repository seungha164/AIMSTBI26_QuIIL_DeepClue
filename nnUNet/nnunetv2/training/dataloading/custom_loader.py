import os
import numpy as np
import warnings
from typing import Union, Tuple, List
from threadpoolctl import threadpool_limits
import torch

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetBaseDataset
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd


class InstanceFairDataLoader(nnUNetDataLoader):
    def __init__(self,
                 data: nnUNetBaseDataset,
                 batch_size: int,
                 patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 label_manager: LabelManager,
                 oversample_foreground_percent: float = 0.0,
                 sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 pad_sides: Union[List[int], Tuple[int, ...]] = None,
                 probabilistic_oversampling: bool = False,
                 transforms=None):
        super().__init__(data, batch_size, patch_size, final_patch_size, label_manager,
                         oversample_foreground_percent, sampling_probabilities, pad_sides,
                         probabilistic_oversampling, transforms)
        print('[HI] InstanceFairDataLoader initialized. This dataloader will sample patches based on instance locations to ensure fair representation of instances during training.')
        
    # ! get_bbox에 instance_locations를 넘겨주기
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate output tensors in final patch size and write transformed samples directly
        data_all = None
        seg_all = None

        with torch.no_grad():
            with threadpool_limits(limits=1, user_api=None):
                for j, i in enumerate(selected_keys):
                    # oversampling foreground will improve stability of model training, especially if many patches are empty
                    # (Lung for example)
                    force_fg = self.get_do_oversample(j)

                    data, seg, seg_prev, properties = self._data.load_case(i)

                    # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
                    # self._data.load_case(i) (see nnUNetDataset.load_case)
                    shape = data.shape[1:]
                    # ! 수정 부분
                    # 사전에 전처리된 properties 딕셔너리에 'instance_locations'를 추가해두었다고 가정합니다.
                    instance_locs = properties.get('instance_locations', None)
                    bbox_lbs, bbox_ubs = self.get_bbox(
                        shape, force_fg, properties['class_locations'], 
                        instance_locations=instance_locs  # ! instance_locations를 get_bbox에 전달
                    )
                    bbox = [[i, j] for i, j in zip(bbox_lbs, bbox_ubs)]

                    data_cropped = torch.from_numpy(crop_and_pad_nd(data, bbox, 0)).float()
                    seg_cropped = torch.from_numpy(crop_and_pad_nd(seg, bbox, -1, cast_cropped_to=np.int16)).to(torch.int16)
                    if seg_prev is not None:
                        seg_prev_cropped = torch.from_numpy(crop_and_pad_nd(seg_prev, bbox, -1, cast_cropped_to=np.int16)).to(torch.int16)
                        seg_cropped = torch.cat((seg_cropped, seg_prev_cropped[None]), dim=0)

                    if self.patch_size_was_2d:
                        data_cropped = data_cropped[:, 0]
                        seg_cropped = seg_cropped[:, 0]

                    if self.transforms is not None:
                        transformed = self.transforms(**{'image': data_cropped, 'segmentation': seg_cropped})
                        data_sample = transformed['image']
                        seg_sample = transformed['segmentation']
                    else:
                        data_sample = data_cropped
                        seg_sample = seg_cropped

                    if data_all is None:
                        data_all = torch.empty((self.batch_size, *data_sample.shape), dtype=torch.float32)
                    data_all[j] = data_sample

                    if isinstance(seg_sample, list):
                        if seg_all is None:
                            seg_all = [torch.empty((self.batch_size, *s.shape), dtype=s.dtype) for s in seg_sample]
                        for s_idx, s in enumerate(seg_sample):
                            seg_all[s_idx][j] = s
                    else:
                        if seg_all is None:
                            seg_all = torch.empty((self.batch_size, *seg_sample.shape), dtype=seg_sample.dtype)
                        seg_all[j] = seg_sample
        return {'data': data_all, 'target': seg_all, 'keys': selected_keys}
    

    def get_bbox(self, data_shape: np.ndarray, force_fg: bool, class_locations: Union[dict, None],
                 overwrite_class: Union[int, Tuple[int, ...]] = None, verbose: bool = False, instance_locations: Union[dict, None] = None):
        # in dataloader 2d we need to select the slice prior to this and also modify the class_locations to only have
        # locations for the given slice
        need_to_pad = self.need_to_pad.copy()
        dim = len(data_shape)

        for d in range(dim):
            # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
            # always
            if need_to_pad[d] + data_shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - data_shape[d]

        # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
        # define what the upper and lower bound can be to then sample form them with np.random.randint
        lbs = [- need_to_pad[i] // 2 for i in range(dim)]
        ubs = [data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i] for i in range(dim)]

        # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
        # at least one of the foreground classes in the patch
        if not force_fg and not self.has_ignore:
            bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
            # print('I want a random location')
        else:
            if not force_fg and self.has_ignore:
                selected_class = self.annotated_classes_key
                if len(class_locations[selected_class]) == 0:
                    # no annotated pixels in this case. Not good. But we can hardly skip it here
                    warnings.warn('Warning! No annotated pixels in image!')
                    selected_class = None
            elif force_fg:
                assert class_locations is not None, 'if force_fg is set class_locations cannot be None'
                if overwrite_class is not None:
                    assert overwrite_class in class_locations.keys(), 'desired class ("overwrite_class") does not ' \
                                                                      'have class_locations (missing key)'
                # this saves us a np.unique. Preprocessing already did that for all cases. Neat.
                # class_locations keys can also be tuple
                eligible_classes_or_regions = [i for i in class_locations.keys() if len(class_locations[i]) > 0]

                # if we have annotated_classes_key locations and other classes are present, remove the annotated_classes_key from the list
                # strange formulation needed to circumvent
                # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
                tmp = [i == self.annotated_classes_key if isinstance(i, tuple) else False for i in eligible_classes_or_regions]
                if any(tmp):
                    if len(eligible_classes_or_regions) > 1:
                        eligible_classes_or_regions.pop(np.where(tmp)[0][0])

                if len(eligible_classes_or_regions) == 0:
                    # this only happens if some image does not contain foreground voxels at all
                    selected_class = None
                    if verbose:
                        print('case does not contain any foreground classes')
                else:
                    # I hate myself. Future me aint gonna be happy to read this
                    # 2022_11_25: had to read it today. Wasn't too bad
                    # 1. 존재하는 전경 클래스 중 하나를 무작위로 선택
                    selected_class = eligible_classes_or_regions[np.random.choice(len(eligible_classes_or_regions))] if \
                        (overwrite_class is None or (overwrite_class not in eligible_classes_or_regions)) else overwrite_class
                # print(f'I want to have foreground, selected class: {selected_class}')
            else:
                raise RuntimeError('lol what!?')

            if selected_class is not None:
                #* 🌟 [핵심 변경 사항: Instance-aware Sampling] 🌟
                # 1. 해당 클래스의 '병변 덩어리(Instance) 목록'을 가져옵니다.
                instances = instance_locations[selected_class]
                # 2. 작은 병변이든 큰 병변이든 1/N 확률로 병변 덩어리를 1개 선택합니다.
                selected_instance = instances[np.random.choice(len(instances))]
                # 3. 선택된 병변 내부의 복셀 중 하나를 패치의 중심으로 삼습니다.
                selected_voxel = selected_instance[np.random.choice(len(selected_instance))]
                # # 2. 선택된 클래스에 해당하는 "모든 복셀(픽셀)의 좌표 리스트"를 가져옴
                # voxels_of_that_class = class_locations[selected_class]
                # # 3. 그 중 "단 하나의 복셀"을 무작위로 선택 (패치의 중심점이 될 복셀)
                # #### Volume Bias 발생
                # selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]

                #* 4. 선택된 복셀이 패치의 정중앙에 오도록 Bounding Box의 Lower Bound를 계산
                # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
                # Make sure it is within the bounds of lb and ub
                # i + 1 because we have first dimension 0!
                bbox_lbs = [max(lbs[i], selected_voxel[i + 1] - self.patch_size[i] // 2) for i in range(dim)]
            else:
                # If the image does not contain any foreground classes, we fall back to random cropping
                bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]

        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]

        return bbox_lbs, bbox_ubs