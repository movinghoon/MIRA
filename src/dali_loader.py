import os
import math
from pathlib import Path
from typing import Callable, Iterable, Union

# nvidia-dali
from nvidia.dali import fn, ops, types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import LastBatchPolicy
from nvidia.dali.plugin.pytorch import DALIGenericIterator as _DALIGenericIterator


# DALIGenericIterator
class DALIGenericIterator(_DALIGenericIterator):
    def __len__(self):
        size = self._size_no_pad // self._shards_num if self._last_batch_policy == LastBatchPolicy.DROP else self.size
        if self._last_batch_policy != LastBatchPolicy.DROP:
            return math.ceil(size / self.batch_size) if self._reader_name else math.ceil(size / (self._num_gpus * self.batch_size))
        else:
            return size // self.batch_size if self._reader_name else size // (self._num_gpus * self.batch_size)

    def __next__(self):
        batch = super().__next__()[0]
        *all_x, targets = [batch[v] for v in self.output_map]
        targets = targets.squeeze(-1).long().detach().clone()
        all_x = [x.detach().clone() for x in all_x]
        return all_x, targets


# Augmentation Functions
class Mux:
    def __init__(self, prob: float):
        self.to_bool = ops.Cast(dtype=types.DALIDataType.BOOL)
        self.rng = ops.random.CoinFlip(probability=prob)

    def __call__(self, true_case, false_case):
        condition = self.to_bool(self.rng())
        neg_condition = condition ^ True
        return condition * true_case + neg_condition * false_case


class RandomGrayScaleConversion:
    def __init__(self, prob: float = 0.2, device: str = "gpu"):
        self.mux = Mux(prob=prob)
        self.grayscale = ops.ColorSpaceConversion(
            device=device, image_type=types.RGB, output_type=types.GRAY
        )

    def __call__(self, images):
        out = self.grayscale(images)
        out = fn.cat(out, out, out, axis=2)
        return self.mux(true_case=out, false_case=images)


class RandomColorJitter:
    def __init__(
            self,
            brightness: float,
            contrast: float,
            saturation: float,
            hue: float,
            prob: float = 0.8,
            device: str = "gpu",
    ):
        assert 0 <= hue <= 0.5
        self.mux = Mux(prob=prob)
        self.color = ops.ColorTwist(device=device)

        # look at torchvision docs to see how colorjitter samples stuff
        # for bright, cont and sat, it samples from [1-v, 1+v]
        # for hue, it samples from [-hue, hue]
        self.brightness = 1
        self.contrast = 1
        self.saturation = 1
        self.hue = 0

        if brightness:
            self.brightness = ops.random.Uniform(range=[max(0, 1 - brightness), 1 + brightness])

        if contrast:
            self.contrast = ops.random.Uniform(range=[max(0, 1 - contrast), 1 + contrast])

        if saturation:
            self.saturation = ops.random.Uniform(range=[max(0, 1 - saturation), 1 + saturation])

        if hue:
            # dali uses hue in degrees for some reason...
            hue = 360 * hue
            self.hue = ops.random.Uniform(range=[-hue, hue])

    def __call__(self, images):
        out = self.color(
            images,
            brightness=self.brightness() if callable(self.brightness) else self.brightness,
            contrast=self.contrast() if callable(self.contrast) else self.contrast,
            saturation=self.saturation() if callable(self.saturation) else self.saturation,
            hue=self.hue() if callable(self.hue) else self.hue,
        )
        return self.mux(true_case=out, false_case=images)


class RandomGaussianBlur:
    def __init__(self, prob: float = 0.5, window_size: int = 23, device: str = "gpu"):
        self.mux = Mux(prob=prob)
        # gaussian blur
        self.gaussian_blur = ops.GaussianBlur(device=device, window_size=(window_size, window_size))
        self.sigma = ops.random.Uniform(range=[0, 1])

    def __call__(self, images):
        sigma = self.sigma() * 1.9 + 0.1
        out = self.gaussian_blur(images, sigma=sigma)
        return self.mux(true_case=out, false_case=images)


class RandomSolarize:
    def __init__(self, threshold: int = 128, prob: float = 0.0):
        self.mux = Mux(prob=prob)
        self.threshold = threshold

    def __call__(self, images):
        inverted_img = 255 - images
        mask = images >= self.threshold
        out = mask * inverted_img + (True ^ mask) * images
        return self.mux(true_case=out, false_case=images)


# Transform
class ImagenetTransform:
    def __init__(
            self,
            device: str,
            brightness: float,
            contrast: float,
            saturation: float,
            hue: float,
            gray_prob: float = 0.2,
            gaussian_prob: float = 0.5,
            solarization_prob: float = 0.0,
            size: int = 224,
            min_scale: float = 0.08,
            max_scale: float = 1.0,
    ):
        # random crop
        self.random_crop = ops.RandomResizedCrop(
            device=device,
            size=size,
            random_area=(min_scale, max_scale),
            interp_type=types.INTERP_CUBIC,
        )

        # color jitter
        self.random_color_jitter = RandomColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            prob=0.8,
            device=device,
        )

        # grayscale conversion
        self.random_grayscale = RandomGrayScaleConversion(prob=gray_prob, device=device)

        # gaussian blur
        self.random_gaussian_blur = RandomGaussianBlur(prob=gaussian_prob, device=device)

        # solarization
        self.random_solarization = RandomSolarize(prob=solarization_prob)

        # normalize and horizontal flip
        self.cmn = ops.CropMirrorNormalize(
            device=device,
            dtype=types.FLOAT,
            output_layout=types.NCHW,
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.228 * 255, 0.224 * 255, 0.225 * 255],
        )
        self.coin05 = ops.random.CoinFlip(probability=0.5)

        self.str = (
            "ImagenetTransform("
            f"random_crop({min_scale}, {max_scale}), "
            f"random_color_jitter(brightness={brightness}, "
            f"contrast={contrast}, saturation={saturation}, hue={hue}), "
            f"random_gray_scale, random_gaussian_blur({gaussian_prob}), "
            f"random_solarization({solarization_prob}), "
            "crop_mirror_resize())"
        )

    def __str__(self) -> str:
        return self.str

    def __call__(self, images):
        out = self.random_crop(images)
        out = self.random_color_jitter(out)
        out = self.random_grayscale(out)
        out = self.random_gaussian_blur(out)
        out = self.random_solarization(out)
        out = self.cmn(out, mirror=self.coin05())
        return out


# Pipeline
class PretrainPipeline(Pipeline):
    def __init__(
            self,
            data_path: Union[str, Path],
            batch_size: int,
            device: str,
            transforms: Union[Callable, Iterable],
            random_shuffle: bool = True,
            device_id: int = 0,
            shard_id: int = 0,
            num_shards: int = 1,
            num_threads: int = 4,
            seed: int = 12,
    ):
        seed += device_id
        super().__init__(
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            seed=seed,
            # prefetch_queue_depth=1,
        )
        # device
        self.device = device

        # reader
        data_path = Path(data_path)
        self.reader = ops.readers.File(
            file_root=data_path,
            shard_id=shard_id,
            num_shards=num_shards,
            shuffle_after_epoch=random_shuffle,
        )

        # decoder
        decoder_device = "mixed" if self.device == "gpu" else "cpu"
        device_memory_padding = 211025920 if decoder_device == "mixed" else 0
        host_memory_padding = 140544512 if decoder_device == "mixed" else 0
        self.decode = ops.decoders.Image(
            device=decoder_device,
            output_type=types.RGB,
            device_memory_padding=device_memory_padding,
            host_memory_padding=host_memory_padding,
        )
        self.to_int64 = ops.Cast(dtype=types.INT64, device=device)

        # transformations
        self.transforms = transforms

    def define_graph(self):
        # read images from memory
        inputs, labels = self.reader(name="Reader")
        
        # augmentations
        images = self.decode(inputs)
        crops = [transform(images) for transform in self.transforms]

        # labels
        if self.device == "gpu":
            labels = labels.gpu()
        labels = self.to_int64(labels)
        return *crops, labels


class ClassificationPipeline(Pipeline):
    def __init__(
        self,
        data_path: str,
        batch_size: int,
        device: str,
        validation: bool = False,
        device_id: int = 0,
        shard_id: int = 0,
        num_shards: int = 1,
        num_threads: int = 4,
        seed: int = 12,
    ):
        seed += device_id
        super().__init__(batch_size, num_threads, device_id, seed)

        self.device = device
        self.validation = validation

        self.reader = ops.readers.File(
            file_root=data_path,
            shard_id=shard_id,
            num_shards=num_shards,
            shuffle_after_epoch=not self.validation,
        )
        decoder_device = "mixed" if self.device == "gpu" else "cpu"
        # ask nvJPEG to preallocate memory for the biggest sample in ImageNet for CPU and GPU to avoid reallocations in runtime
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
        preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
        preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0
        self.decode = ops.decoders.Image(
            device=decoder_device,
            output_type=types.RGB,
            device_memory_padding=device_memory_padding,
            host_memory_padding=host_memory_padding,
            preallocate_width_hint=preallocate_width_hint,
            preallocate_height_hint=preallocate_height_hint,
        )

        # crop operations
        if self.validation:
            self.resize = ops.Resize(
                device=self.device,
                resize_shorter=256,
                interp_type=types.INTERP_CUBIC,
            )
            # center crop and normalize
            self.cmn = ops.CropMirrorNormalize(
                device=self.device,
                dtype=types.FLOAT,
                output_layout=types.NCHW,
                crop=(224, 224),
                mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                std=[0.228 * 255, 0.224 * 255, 0.225 * 255],
            )
        else:
            self.resize = ops.RandomResizedCrop(
                device=self.device,
                size=224,
                random_area=(0.08, 1.0),
                interp_type=types.INTERP_LINEAR,    # types.INTERP_CUBIC,
            )
            # normalize and horizontal flip
            self.cmn = ops.CropMirrorNormalize(
                device=self.device,
                dtype=types.FLOAT,
                output_layout=types.NCHW,
                mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                std=[0.228 * 255, 0.224 * 255, 0.225 * 255],
            )

        self.coin05 = ops.random.CoinFlip(probability=0.5)
        self.to_int64 = ops.Cast(dtype=types.INT64, device=device)

    def define_graph(self):
        """Defines the computational graph for dali operations."""

        # read images from memory
        inputs, labels = self.reader(name="Reader")
        images = self.decode(inputs)

        # crop into large and small images
        images = self.resize(images)

        if self.validation:
            # crop and normalize
            images = self.cmn(images)
        else:
            # normalize and maybe apply horizontal flip with 0.5 chance
            images = self.cmn(images, mirror=self.coin05())

        if self.device == "gpu":
            labels = labels.gpu()
        # PyTorch expects labels as INT64
        labels = self.to_int64(labels)

        return images, labels


# Loader
def dali_loader(data_dir,
                batch_size,
                local_rank,
                num_gpus,
                num_workers,
                seed,
                dali_device='gpu'):
    # transforms -- DINO Transform
    """
        # global aug1
        global_aug1 = transforms.Compose([
            transforms.RandomResizedCrop(sizes[0], scale=(scale, 1.), interpolation=Image.BICUBIC),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # global aug2
        global_aug2 = transforms.Compose([
            transforms.RandomResizedCrop(sizes[0], scale=(scale, 1.), interpolation=Image.BICUBIC),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
            transforms.RandomApply([Solarize()], p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    """
    aug1 = ImagenetTransform(device=dali_device,
                             brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1,
                             gray_prob=0.2,
                             gaussian_prob=1.0,
                             solarization_prob=0.,
                             size=224, min_scale=0.14, max_scale=1.0)
    aug2 = ImagenetTransform(device=dali_device,
                             brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1,
                             gray_prob=0.2,
                             gaussian_prob=0.1,
                             solarization_prob=0.2,
                             size=224, min_scale=0.14, max_scale=1.0)

    # pipline
    pipeline = PretrainPipeline(data_path=os.path.join(data_dir, 'train'),
                                batch_size=batch_size,
                                transforms=[aug1, aug2],
                                random_shuffle=True,
                                device=dali_device,
                                device_id=local_rank,
                                shard_id=local_rank,
                                num_shards=num_gpus,
                                num_threads=num_workers,
                                seed=seed)
    
    # data-loader
    train_loader = DALIGenericIterator(
        pipelines=[pipeline],
        output_map=["view1", "view2", "label"],
        reader_name="Reader",
        last_batch_policy=LastBatchPolicy.DROP,
        auto_reset=True,
    )
    return train_loader


def dali_classification_loader(data_dir,
                               batch_size,
                               local_rank,
                               num_gpus,
                               num_workers,
                               seed,
                               dali_device='gpu'):
    # pipeline
    pipeline = ClassificationPipeline(data_path=data_dir,
                                      batch_size=batch_size,
                                      device=dali_device,
                                      validation=False,
                                      device_id=local_rank,
                                      shard_id=local_rank,
                                      num_shards=num_gpus,
                                      num_threads=num_workers,
                                      seed=seed)
    
    # data-loader
    train_loader = DALIGenericIterator(
        pipelines=[pipeline],
        output_map=["x", "label"],
        reader_name="Reader",
        last_batch_policy=LastBatchPolicy.DROP,
        auto_reset=True,
    )
    return train_loader
