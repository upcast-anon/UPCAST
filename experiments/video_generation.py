from datasets.video import (
    MinecraftAdvancedVideoDataset,
    RealEstate10KAdvancedVideoDataset,
    RealEstate10KMiniAdvancedVideoDataset,
    ARKitScenesAdvancedVideoDataset,
)
from algorithms.dfot import (
    DFoTUPCAST,
    DFoTRepresentationAlignment,
    DFoTActionUPCAST,
    DFoTVideo,
    DFoTVideoPose,
)
from .base_exp import BaseLightningExperiment
from .data_modules.utils import _data_module_cls


class VideoGenerationExperiment(BaseLightningExperiment):
    """
    A video generation experiment
    """

    compatible_algorithms = dict(
        dfot_video=DFoTVideo,
        dfot_video_pose=DFoTVideoPose,
        dfot_upcast=DFoTUPCAST,
        dfot_representation_alignment=DFoTRepresentationAlignment,
        dfot_action_upcast=DFoTActionUPCAST,
    )

    compatible_datasets = dict(
        # video datasets
        minecraft=MinecraftAdvancedVideoDataset,
        realestate10k=RealEstate10KAdvancedVideoDataset,
        realestate10k_mini=RealEstate10KMiniAdvancedVideoDataset,
        arkitscenes_eval=ARKitScenesAdvancedVideoDataset,
    )

    data_module_cls = _data_module_cls
