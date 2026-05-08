



__version__ = "8.3.74"



import os





if not os.environ.get("OMP_NUM_THREADS"):

    os.environ["OMP_NUM_THREADS"] = "1"



from ultralytics.models import NAS, RTDETR, SAM, YOLO, FastSAM, YOLOWorld

from ultralytics.utils import ASSETS, SETTINGS

from ultralytics.utils.checks import check_yolo as checks

from ultralytics.utils.downloads import download



settings = SETTINGS

__all__ = (

    "__version__",

    "ASSETS",

    "YOLO",

    "YOLOWorld",

    "NAS",

    "SAM",

    "FastSAM",

    "RTDETR",

    "checks",

    "download",

    "settings",

)

