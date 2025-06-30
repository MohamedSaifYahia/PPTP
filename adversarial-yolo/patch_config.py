# # from torch import optim


# # class BaseConfig(object):
# #     """
# #     Default parameters for all config files.
# #     """

# #     def __init__(self):
# #         """
# #         Set the defaults.
# #         """
# #         self.img_dir = "inria/Train/pos"
# #         self.lab_dir = "inria/Train/pos/yolo-labels"
# #         self.cfgfile = "cfg/yolo.cfg"
# #         self.weightfile = "weights/yolo.weights"
# #         self.printfile = "non_printability/30values.txt"
# #         self.patch_size = 300

# #         self.start_learning_rate = 0.03

# #         self.patch_name = 'base'

# #         self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50)
# #         self.max_tv = 0

# #         self.batch_size = 20

# #         self.loss_target = lambda obj, cls: obj * cls


# # class Experiment1(BaseConfig):
# #     """
# #     Model that uses a maximum total variation, tv cannot go below this point.
# #     """

# #     def __init__(self):
# #         """
# #         Change stuff...
# #         """
# #         super().__init__()

# #         self.patch_name = 'Experiment1'
# #         self.max_tv = 0.165


# # class Experiment2HighRes(Experiment1):
# #     """
# #     Higher res
# #     """

# #     def __init__(self):
# #         """
# #         Change stuff...
# #         """
# #         super().__init__()

# #         self.max_tv = 0.165
# #         self.patch_size = 400
# #         self.patch_name = 'Exp2HighRes'

# # class Experiment3LowRes(Experiment1):
# #     """
# #     Lower res
# #     """

# #     def __init__(self):
# #         """
# #         Change stuff...
# #         """
# #         super().__init__()

# #         self.max_tv = 0.165
# #         self.patch_size = 100
# #         self.patch_name = "Exp3LowRes"

# # class Experiment4ClassOnly(Experiment1):
# #     """
# #     Only minimise class score.
# #     """

# #     def __init__(self):
# #         """
# #         Change stuff...
# #         """
# #         super().__init__()

# #         self.patch_name = 'Experiment4ClassOnly'
# #         self.loss_target = lambda obj, cls: cls




# # class Experiment1Desktop(Experiment1):
# #     """
# #     """

# #     def __init__(self):
# #         """
# #         Change batch size.
# #         """
# #         super().__init__()

# #         self.batch_size = 8
# #         self.patch_size = 400


# # class ReproducePaperObj(BaseConfig):
# #     """
# #     Reproduce the results from the paper: Generate a patch that minimises object score.
# #     """

# #     def __init__(self):
# #         super().__init__()

# #         self.batch_size = 8
# #         self.patch_size = 300

# #         self.patch_name = 'ObjectOnlyPaper'
# #         self.max_tv = 0.165

# #         self.loss_target = lambda obj, cls: obj


# # patch_configs = {
# #     "base": BaseConfig,
# #     "exp1": Experiment1,
# #     "exp1_des": Experiment1Desktop,
# #     "exp2_high_res": Experiment2HighRes,
# #     "exp3_low_res": Experiment3LowRes,
# #     "exp4_class_only": Experiment4ClassOnly,
# #     "paper_obj": ReproducePaperObj
# # }

# # from torch import optim

# # class BaseConfig(object):
# #     """
# #     Base configuration for adversarial patch generation against YOLOv5.
# #     """
# #     def __init__(self):
# #         """
# #         Set the default parameters.
# #         """
# #         # --- Paths ---
# #         self.img_dir = "inria/Train/pos"
# #         self.lab_dir = "inria/Train/pos/yolo-labels"
# #         self.printfile = "non_printability/30values.txt"

# #         # UPDATED: Path to the YOLOv5 weights file. Assumes yolov5s.pt is in the parent directory.
# #         self.weightfile = "../yolov5s.pt"

# #         # REMOVED: self.cfgfile is no longer needed for YOLOv5.

# #         # --- Patch & Training Parameters ---
# #         self.patch_size = 300
# #         self.batch_size = 20
# #         self.start_learning_rate = 0.03

# #         # --- Scheduler ---
# #         # This function creates the learning rate scheduler
# #         self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50)
        
# #         # --- Naming ---
# #         self.patch_name = 'base_v5'
        
# #         # REMOVED: self.loss_target is obsolete. The new training script directly minimizes
# #         # the objectness score from the YOLOv5 output.
# #         # REMOVED: self.max_tv is not used in the new simplified loss calculation.

# # class PaperReproduction(BaseConfig):
# #     """
# #     Aims to reproduce the settings from the original paper (e.g., batch size) 
# #     but targets a YOLOv5 model. The attack minimizes the objectness score.
# #     """
# #     def __init__(self):
# #         super().__init__()

# #         self.patch_name = 'paper_repro_v5'
# #         self.batch_size = 8
# #         self.patch_size = 300


# # class HighResPatch(BaseConfig):
# #     """
# #     Configuration for generating a higher resolution patch.
# #     """
# #     def __init__(self):
# #         super().__init__()

# #         self.patch_name = 'high_res_400_v5'
# #         self.patch_size = 400
# #         self.batch_size = 16 # Lower batch size for larger patch to conserve memory


# # class LowResPatch(BaseConfig):
# #     """
# #     Configuration for generating a lower resolution patch.
# #     """
# #     def __init__(self):
# #         super().__init__()

# #         self.patch_name = "low_res_150_v5"
# #         self.patch_size = 150
# #         self.batch_size = 24


# # # The dictionary linking command-line arguments to the config classes
# # patch_configs = {
# #     "default": BaseConfig,
# #     "paper_obj": PaperReproduction,
# #     "high_res": HighResPatch,
# #     "low_res": LowResPatch
# # }
# # patch_config.py

# #jajajajajaja
# from torch import optim

# class BaseConfig(object):
#     """
#     Base configuration for adversarial patch generation against YOLOv5.
#     """
#     def __init__(self):
#         """
#         Set the default parameters.
#         """
#         # --- Paths ---
#         self.img_dir = "inria/Train/pos"
#         self.lab_dir = "inria/Train/pos/yolo-labels"
#         self.printfile = "non_printability/30values.txt"
#         self.weightfile = "../yolov5s.pt"

#         # --- Patch & Training Parameters ---
#         self.patch_size = 300
#         self.batch_size = 20
#         self.start_learning_rate = 0.03

#         # --- NEW: Define the input image size for the model ---
#         # YOLOv5 standard is 640. This makes our code independent of model API changes.
#         self.image_size = 640

#         # --- Scheduler ---
#         self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50)
        
#         # --- Naming ---
#         self.patch_name = 'base_v5'

# class PaperReproduction(BaseConfig):
#     """
#     Aims to reproduce the settings from the original paper (e.g., batch size) 
#     but targets a YOLOv5 model. The attack minimizes the objectness score.
#     """
#     def __init__(self):
#         super().__init__()

#         self.patch_name = 'paper_repro_v5'
#         self.batch_size = 8
#         self.patch_size = 300

# class HighResPatch(BaseConfig):
#     """
#     Configuration for generating a higher resolution patch.
#     """
#     def __init__(self):
#         super().__init__()

#         self.patch_name = 'high_res_400_v5'
#         self.patch_size = 400
#         self.batch_size = 16

# class LowResPatch(BaseConfig):
#     """
#     Configuration for generating a lower resolution patch.
#     """
#     def __init__(self):
#         super().__init__()

#         self.patch_name = "low_res_150_v5"
#         self.patch_size = 150
#         self.batch_size = 24

# patch_configs = {
#     "default": BaseConfig,
#     "paper_obj": PaperReproduction,
#     "high_res": HighResPatch,
#     "low_res": LowResPatch
# }

from torch import optim

class BaseConfig(object):
    """
    Base configuration for adversarial patch generation against YOLOv5.
    """
    def __init__(self):
        # --- Paths ---
        self.img_dir = "inria/Train/pos"
        self.lab_dir = "inria/Train/pos/yolo-labels"
        self.printfile = "non_printability/30values.txt"
        self.weightfile = "../yolov5s.pt"

        # --- Model & Patch Parameters ---
        self.image_size = 640
        self.patch_size = 300
        
        # --- Training Parameters ---
        self.batch_size = 8
        self.start_learning_rate = 0.03
        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50)
        
        # --- Loss Weights (for easy tuning) ---
        self.nps_weight = 0.01
        self.tv_weight = 2.5
        
        # --- Naming ---
        self.patch_name = 'repro_v5'

# The dictionary for command-line arguments. We only need one for now.
patch_configs = {
    "paper_obj": BaseConfig
}