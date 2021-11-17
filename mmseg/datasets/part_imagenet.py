from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class PartImageNetDataset(CustomDataset):
    """PartImageNet dataset.

    In segmentation map annotation for PartImageNet, Train-IDs are from 0 to 39, 
    they are all 40 part categories. 255 is the ignore index (unlabeled data).  
    The ``img_suffix`` is fixed to '.JPEG',
    and ``seg_map_suffix`` is fixed to '.png'.
    """
    CLASSES = ('Mammal Head', 'Mammal Body', 'Mammal Foot', 'Mammal Tail', 'Monkey Head', 
               'Monkey Body', 'Monkey Hand', 'Monkey Foot', 'Monkey Tail', 'Fish Head',
               'Fish Body', 'Fish Fin', 'Fish Tail', 'Bird Head', 'Bird Body', 
               'Bird Wing','Bird Foot', 'Bird Tail', 'Snake Head', 'Snake Body', 
               'Reptile Head', 'Reptile Body', 'Reptile Foot', 'Reptile Tail', 'Motor Vehicle Body',
               'Motor Vehicle Tier', 'Motor Vehicle Side Mirror', 'Non-Motor Vehicle Body',
               'Non-Motor Vehicle Head', 'Non-Motor Vehicle Seat', 'Non-Motor Vehicle Tier',
               'Boat Body', 'Boat Sail', 'Plane Head', 'Plane Body', 'Plane Engine',
               'Plane Wing', 'Plane Tail', 'Bottle Mouth', 'Bottle Body','Background')

    PALETTE = [[ 78, 178,  93], [ 34, 178, 138], [ 19, 178, 152], [178, 160,   0], [  0,   0, 172], 
               [  0, 130, 178], [130,   0,   0], [178, 143,   0], [ 89,   0,   0], [  0,  20, 178], 
               [167, 177,   5], [  0,  75, 178], [  0, 148, 178], [178,  76,   0], [178,  25,   0], 
               [  0,  57, 178], [ 49, 178, 123], [152, 178,  19], [  0,   0,  89], [178,  42,   0], 
               [  0,   2, 178], [  0, 112, 178], [178,   0, 110], [178, 110,   0], [108, 178,  64], 
               [  0,   0, 130], [178,  93,   0], [151,   0,   0], [  0,   0, 151], [172,   8,   0], 
               [138, 178,  34], [178,  59,   0], [  0,  93, 178], [123, 178,  49], [ 64, 178, 108], 
               [ 93, 178,  78], [  0,  38, 178], [  5, 167, 167], [178, 126,   0], [110,   0,   0],
                 [0,   0,   0]]

    def __init__(self, **kwargs):
        super(PartImageNetDataset, self).__init__(
            img_suffix='.JPEG', seg_map_suffix='_labelTrainIds.png', **kwargs)
