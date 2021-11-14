import numpy as np
from pycocotools import mask
from PIL import Image, ImagePalette # For indexed images
import matplotlib # For Matlab's color maps
from pycocotools.coco import COCO
import os
import shutil

CONFLICT_PIXEL_NUM = 0


def getCMap(stuffStartId=92, stuffEndId=182, cmapName='jet', addThings=True, addUnlabeled=True, addOther=True):
    '''
    Create a color map for the classes in the COCO Stuff Segmentation Challenge.
    :param stuffStartId: (optional) index where stuff classes start
    :param stuffEndId: (optional) index where stuff classes end
    :param cmapName: (optional) Matlab's name of the color map
    :param addThings: (optional) whether to add a color for the 91 thing classes
    :param addUnlabeled: (optional) whether to add a color for the 'unlabeled' class
    :param addOther: (optional) whether to add a color for the 'other' class
    :return: cmap - [c, 3] a color map for c colors where the columns indicate the RGB values
    '''

    # Get jet color map from Matlab
    labelCount = stuffEndId - stuffStartId + 1
    cmapGen = matplotlib.cm.get_cmap(cmapName, labelCount)
    cmap = cmapGen(np.arange(labelCount))
    cmap = cmap[:, 0:3]

    # Reduce value/brightness of stuff colors (easier in HSV format)
    cmap = cmap.reshape((-1, 1, 3))
    hsv = matplotlib.colors.rgb_to_hsv(cmap)
    hsv[:, 0, 2] = hsv[:, 0, 2] * 0.7
    cmap = matplotlib.colors.hsv_to_rgb(hsv)
    cmap = cmap.reshape((-1, 3))

    # Permute entries to avoid classes with similar name having similar colors
    st0 = np.random.get_state()
    np.random.seed(42)
    perm = np.random.permutation(labelCount)
    np.random.set_state(st0)
    cmap = cmap[perm, :]

    # Add black (or any other) color for each thing class
    if addThings:
        thingsPadding = np.zeros((stuffStartId - 1, 3))
        cmap = np.vstack((thingsPadding, cmap))

    # Add black color for 'unlabeled' class
    if addUnlabeled:
        cmap = np.vstack((cmap, (1.0, 1.0, 1.0)))

    # Add yellow/orange color for 'other' class
    if addOther:
        cmap = np.vstack((cmap, (1.0, 0.843, 0.0)))

    return cmap


def cocoSegmentationToSegmentationMap(coco, imgId, checkUniquePixelLabel=True, includeCrowd=False):
    '''
    Convert COCO GT or results for a single image to a segmentation map. 255 indicates unlabeled class.
    :param coco: an instance of the COCO API (ground-truth or result)
    :param imgId: the id of the COCO image
    :param checkUniquePixelLabel: (optional) whether every pixel can have at most one label
    :param includeCrowd: whether to include 'crowd' thing annotations as 'other' (or void)
    :return: labelMap - [h x w] segmentation map that indicates the label of each pixel
    '''

    # Init
    curImg = coco.imgs[imgId]
    imageSize = (curImg['height'], curImg['width'])
    # labelMap = np.zeros(imageSize)
    labelMap = np.ones(imageSize) * 255

    # Get annotations of the current image (may be empty)
    imgAnnots = [a for a in coco.anns.values() if a['image_id'] == imgId]
    if includeCrowd:
        annIds = coco.getAnnIds(imgIds=imgId)
    else:
        annIds = coco.getAnnIds(imgIds=imgId, iscrowd=False)
    imgAnnots = coco.loadAnns(annIds)

    # Combine all annotations of this image in labelMap
    #labelMasks = mask.decode([a['segmentation'] for a in imgAnnots])
    for a in range(0, len(imgAnnots)):
        # Dealing with a cocoAPI issue when segmentation contains only 4 numbers,
        # it'll be identified as a bounding box not a polygon format.
        # A quick solution is to avoid len(segmentation) == 4 by adding additional points.
        for _segm in imgAnnots[a]['segmentation']:
            while len(_segm) <= 4:
                whatever_offset = -1 if _segm[-1] > 1.0 else 1
                _segm.extend([_segm[-2], _segm[-1] + whatever_offset])

        labelMask = coco.annToMask(imgAnnots[a]) == 1
        #labelMask = labelMasks[:, :, a] == 1
        newLabel = imgAnnots[a]['category_id']

        if checkUniquePixelLabel and (labelMap[labelMask] != 255).any():
            # raise Exception('Error: Some pixels have more than one label (image %d)!' % (imgId))
            print('Error: Some pixels have more than one label (image %d)!' % (imgId))
            global CONFLICT_PIXEL_NUM
            CONFLICT_PIXEL_NUM += 1

        labelMap[labelMask] = newLabel

    return labelMap


def cocoSegmentationToPng(coco, imgId, pngPath, includeCrowd=False, if_color_map=False):
    '''
    Convert COCO GT or results for a single image to a segmentation map and write it to disk.
    :param coco: an instance of the COCO API (ground-truth or result)
    :param imgId: the COCO id of the image (last part of the file name)
    :param pngPath: the path of the .png file
    :param includeCrowd: whether to include 'crowd' thing annotations as 'other' (or void)
    :param if_color_map: whether the segmentation map is colored mode (palette mode of pillow) or not.
    :return: None
    '''

    # Create label map
    labelMap = cocoSegmentationToSegmentationMap(coco, imgId, includeCrowd=includeCrowd)
    labelMap = labelMap.astype(np.int8)

    if if_color_map:
        # Get color map and convert to PIL's format
        # cmap = getCMap()
        cmap = getCMap(stuffStartId=0, stuffEndId=39, cmapName='jet', addThings=False, addUnlabeled=True, addOther=False)  # hard coded for PartImageNet dataset

        cmap = (cmap * 255).astype(int)
        padding = np.zeros((256-cmap.shape[0], 3), np.int8)
        cmap = np.vstack((cmap, padding))
        cmap = cmap.reshape((-1))
        assert len(cmap) == 768, 'Error: Color map must have exactly 256*3 elements!'

        # Write to png file
        png = Image.fromarray(labelMap).convert('P')
        png.putpalette(list(cmap))
    else:
        # convert to grayscale format, the pixel value fall in range [0, num_classes - 1] and 255 indicates unlabeled data.
        png = Image.fromarray(labelMap).convert('L')

    png.save(pngPath, format='PNG')


if __name__ == "__main__":
    DATASET_LEN = {'train': 16540, 'val': 2957, 'test': 4598}
    root_path = '/mnt/data0/xiaoding'
    raw_dataDir = os.path.join(root_path, 'PartImageNet')
    dataDir = os.path.join(root_path, 'PartImageNet_coco_format_test')

    # re-organize file structure
    if os.path.exists(dataDir):
        shutil.rmtree(dataDir)
    os.mkdir(dataDir)
    shutil.copytree(os.path.join(raw_dataDir, 'anno'), os.path.join(dataDir, 'annotations'))
    os.mkdir(os.path.join(dataDir, 'annotations', 'test'))
    os.mkdir(os.path.join(dataDir, 'annotations', 'train'))
    os.mkdir(os.path.join(dataDir, 'annotations', 'val'))

    os.mkdir(os.path.join(dataDir, 'images'))
    for dataType in ['train', 'val', 'test']:
        os.mkdir(os.path.join(dataDir, 'images', dataType))
        dirs = os.listdir(os.path.join(raw_dataDir, dataType))
        for sub_dir in dirs:
            cmd = 'mv {}/* {}/'.format(os.path.join(raw_dataDir, dataType, sub_dir),
                                       os.path.join(dataDir, 'images', dataType))
            os.system(cmd)

    # generate annotation in image format from json

    for dataType in ['train', 'val', 'test']:
        annFile = '{}/annotations/{}.json'.format(dataDir, dataType)
        SavePngDir = '{}/annotations/{}'.format(dataDir, dataType)

        # initialize COCO api for instance annotations
        coco = COCO(annFile)

        # display COCO categories and supercategories
        cats = coco.loadCats(coco.getCatIds())
        nms = [cat['name'] for cat in cats]
        print('COCO categories: \n{}\n'.format(','.join(nms)))

        nms = set([cat['supercategory'] for cat in cats])
        print('COCO supercategories: \n{}'.format(' '.join(nms)))

        for img_id in range(DATASET_LEN[dataType]):
            file_name = os.path.splitext(coco.imgs[img_id]['file_name'])[0]
            png_path = os.path.join(SavePngDir, '{}.png'.format(file_name))

            cocoSegmentationToPng(coco, imgId=img_id, pngPath=png_path)

            # # for debug
            # cocoSegmentationToPng(coco, imgId=img_id, pngPath=png_path, if_color_map=True)