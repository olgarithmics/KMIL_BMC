import cv2
import numpy as np
from openslide import OpenSlideUnsupportedFormatError, ImageSlide
import os
import glob
from PIL import Image
import scipy.io

class WSIOps(object):

    @staticmethod
    def read_wsi(wsi_path):

        try:
            #img_name = "_".join(wsi_path.split("/")[-1].split("_")[:3])
            img_name = os.path.splitext(wsi_path.split("/")[-1])[0]
            wsi_image = ImageSlide(wsi_path)
            level_used = wsi_image.level_count - 1
            rgb_image = np.array(wsi_image.read_region((0, 0), level_used,
                                                       wsi_image.level_dimensions[level_used]))
        except OpenSlideUnsupportedFormatError:
            raise ValueError('Exception: OpenSlideUnsupportedFormatError for %s' % wsi_path)

        return img_name,wsi_image, rgb_image, level_used

    @staticmethod
    def get_image_open(wsi_path):
        try:
            wsi_image = ImageSlide(wsi_path)
            level_used = wsi_image.level_count - 1
            rgb_image = np.array(wsi_image.read_region((0, 0), level_used,
                                                       wsi_image.level_dimensions[level_used]))
        except OpenSlideUnsupportedFormatError:
            raise ValueError('Exception: OpenSlideUnsupportedFormatError for %s' % wsi_path)

        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        lower_red = np.array([20, 20, 20])
        upper_red = np.array([200, 200, 200])

        mask = cv2.inRange(hsv, lower_red, upper_red)

        close_kernel = np.ones((20, 20), dtype=np.uint8)
        image_close = cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel)
        open_kernel = np.ones((5, 5), dtype=np.uint8)
        image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)

        return image_open

    @staticmethod
    def get_bbox(cont_img, rgb_image=None):
        contours,_ = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rgb_contour = None
        if not (rgb_image is None):
            rgb_contour = rgb_image.copy()
            line_color = (255, 0, 0)  # blue color code
            cv2.drawContours(rgb_contour, contours, -1, line_color, 2)
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        return bounding_boxes, rgb_contour

    @staticmethod
    def draw_bbox(image, bounding_boxes):
            rgb_bbox = image.copy()
            for i, bounding_box in enumerate(bounding_boxes):
                x = int(bounding_box[0])
                y = int(bounding_box[1])
                cv2.rectangle(rgb_bbox, (x, y), (x + bounding_box[2], y + bounding_box[3]), color=(0, 0, 255),
                              thickness=2)
            return rgb_bbox


    def hematoxylin_eosin_aug(self,image, low=0.7, high=1.3, seed=None):
        """
        "Quantification of histochemical staining by color deconvolution"
        Arnout C. Ruifrok, Ph.D. and Dennis A. Johnston, Ph.D.
        http://www.math-info.univ-paris5.fr/~lomn/Data/2017/Color/Quantification_of_histochemical_staining.pdf
        Performs random hematoxylin-eosin augmentation
        """
        D = np.array([[1.88, -0.07, -0.60],
                      [-1.02, 1.13, -0.48],
                      [-0.55, -0.13, 1.57]])
        M = np.array([[0.65, 0.70, 0.29],
                      [0.07, 0.99, 0.11],
                      [0.27, 0.57, 0.78]])
        Io = 240

        h, w, c = image.shape
        OD = -np.log10((image.astype("uint16") + 1) / Io)
        C = np.dot(D, OD.reshape(h * w, c).T).T
        r = np.ones(3)
        r[:2] = np.random.RandomState(seed).uniform(low=low, high=high, size=2)
        img_aug = np.dot(C, M) * r

        img_aug = Io * np.exp(-img_aug * np.log(10)) - 1
        img_aug = img_aug.reshape(h, w, c).clip(0, 255).astype("uint8")
        return img_aug


    def normalize_staining(self,image):

        Io = 240
        beta = 0.15
        alpha = 1
        HERef = np.array([[0.5626, 0.2159],
                          [0.7201, 0.8012],
                          [0.4062, 0.5581]])
        maxCRef = np.array([1.9705, 1.0308])

        h, w, c = image.shape
        img = image.reshape(h * w, c)
        OD = -np.log((img.astype("uint16") + 1) / Io)
        ODhat = OD[(OD >= beta).all(axis=1)]
        W, V = np.linalg.eig(np.cov(ODhat, rowvar=False))

        Vec = -V.T[:2][::-1].T
        That = np.dot(ODhat, Vec)
        phi = np.arctan2(That[:, 1], That[:, 0])
        minPhi = np.percentile(phi, alpha)
        maxPhi = np.percentile(phi, 100 - alpha)
        vMin = np.dot(Vec, np.array([np.cos(minPhi), np.sin(minPhi)]))
        vMax = np.dot(Vec, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
        if vMin[0] > vMax[0]:
            HE = np.array([vMin, vMax])
        else:
            HE = np.array([vMax, vMin])

        HE = HE.T
        Y = OD.reshape(h * w, c).T

        C = np.linalg.lstsq(HE, Y,rcond=None)
        maxC = np.percentile(C[0], 99, axis=1)

        C = C[0] / maxC[:, None]
        C = C * maxCRef[:, None]
        Inorm = Io * np.exp(-np.dot(HERef, C))
        Inorm = Inorm.T.reshape(h, w, c).clip(0, 255).astype("uint8")

        return Inorm

    def find_roi_bbox(self, rgb_image):
            rgb_image = cv2.cvtColor(rgb_image, cv2.cv2.COLOR_BGR2RGB)
            norm_image=self.normalize_staining(rgb_image)
            aug_image=self.hematoxylin_eosin_aug(norm_image)

            hsv = cv2.cvtColor(aug_image, cv2.COLOR_BGR2HSV)
            lower_red = np.array([20, 20, 20])
            upper_red = np.array([200, 200, 200])

            mask = cv2.inRange(hsv, lower_red, upper_red)

            close_kernel = np.ones((20, 20), dtype=np.uint8)
            image_close = cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel)
            open_kernel = np.ones((5, 5), dtype=np.uint8)
            image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)

            bounding_boxes, rgb_contour = self.get_bbox(image_open, rgb_image=rgb_image)

            return aug_image,bounding_boxes, rgb_contour, image_open


class PatchExtractor(object):

    def extract_fixed_size_patches_from_WSI(self,
                                            wsi_image, img_name,
                                            level_used, bounding_boxes,
                                            image_open,
                                            patch_save_dir
                                            ):
        mag_factor = pow(2, level_used)
        print('No. of ROIs to extract patches from: %d' % len(bounding_boxes))
        patch_prefix = str("0") if "benign" in img_name else str("1")
        patch_index=0
        #wsi_image = cv2.cvtColor(wsi_image, cv2.cv2.COLOR_RGB2BGRA)
        for bounding_box in bounding_boxes:

            b_x_start = int(bounding_box[0])
            b_y_start = int(bounding_box[1])
            b_x_end = int(bounding_box[0]) + int(bounding_box[2])
            b_y_end = int(bounding_box[1]) + int(bounding_box[3])
            col_cords = np.arange(b_x_start, b_x_end,STEP)
            row_cords = np.arange(b_y_start, b_y_end,STEP)
            if (np.logical_and(len(col_cords), len(row_cords))):
                for row in row_cords:
                    for col in col_cords:
                        mask_gt = image_open[row * mag_factor:row * mag_factor+PATCH_SIZE,col * mag_factor:col * mag_factor+PATCH_SIZE]

                        white_pixel_cnt_gt = cv2.countNonZero(mask_gt)

                        if white_pixel_cnt_gt >= ((PATCH_SIZE * PATCH_SIZE) * 0.85):
                            pil_image = Image.fromarray(wsi_image)
                            ndpi = ImageSlide(pil_image)
                            patch = ndpi.read_region((col * mag_factor, row * mag_factor), 0,
                                                          (PATCH_SIZE, PATCH_SIZE))

                            if not os.path.exists(patch_save_dir):
                                os.mkdir(patch_save_dir)
                                print("Directory ", patch_prefix, " Created ")


                            patch_folder = os.path.join(patch_save_dir,patch_prefix, "img{}".format(img_name))

                            if not os.path.exists(patch_folder):
                                os.makedirs(patch_folder)
                                print("Directory ", patch_folder, " Created ")

                            background = Image.new("RGB", patch.size, (255, 255, 255))
                            background.paste(patch, mask=patch.split()[3])
                            class_name = os.path.basename(img_name).split("_")[2]

                            background.save(os.path.join(patch_folder,
                                                         "img{}-xpos{}-ypos{}-{}.png".format(
                                                         patch_index,int(((col * mag_factor) + (STEP / 2))),
                                                        int(((row * mag_factor) + (STEP / 2))),class_name)))
                            patch_index += 1
                            patch.close()

    def extract_fixed_location_patches(self,
                                    root,
                                    wsi_image, img_name,
                                    patch_save_dir,
                                    patch_size,patch_index):


            PIL_image = Image.fromarray(rgb_image, "RGBA")

            new_image = Image.new("RGB", (PIL_image.size), (255, 255, 255))
            new_image.paste(PIL_image, mask=PIL_image.split()[3])  # 3 is the alpha channel

            aug_image = WSIOps().hematoxylin_eosin_aug(np.array(new_image))

            for enum, cell_type in enumerate(['epithelial', 'fibroblast', 'inflammatory', 'others']):
                dir_cell = img_name + '_' + cell_type + '.mat'

                with open(os.path.join(root, dir_cell), 'rb') as f:
                    mat_cell = scipy.io.loadmat(f)

                for (x, y) in mat_cell['detection']:
                    x = np.round(x)
                    y = np.round(y)

                    if x < np.floor(patch_size / 2):
                        x_start = 0
                        x_end = patch_size
                    elif x > 500 - np.ceil(patch_size / 2):
                        x_start = 500 - patch_size
                        x_end = 500
                    else:
                        x_start = x - np.floor(patch_size / 2)
                        x_end = x + np.ceil(patch_size / 2)
                    if y < np.floor(patch_size / 2):
                        y_start = 0
                        y_end = patch_size
                    elif y > 500 - np.ceil(patch_size / 2):
                        y_start = 500 - patch_size
                        y_end = 500
                    else:
                        y_start = y - np.floor(patch_size / 2)
                        y_end = y + np.ceil(patch_size / 2)

                    patch = aug_image[int(y_start):int(y_end), int(x_start):int(x_end)]

                    patch_folder = os.path.join(patch_save_dir, img_name)

                    if not os.path.exists(patch_folder):
                        os.makedirs(patch_folder)
                        print("Directory ", patch_folder, " Created ")

                    Image.fromarray(patch, "RGB").save(
                        os.path.join(patch_folder, "{}-xpos{}-ypos{}-{}.png".format(
                            img_name, int(x), int(y), cell_type)))

            wsi_image.close()



def extract__patches_from_wsi(wsi_ops, patch_extractor, augmentation=False):
    wsi_paths = glob.glob(os.path.join(WSI_PATH, '*.tif'))
    wsi_paths.sort()
    patch_index=0
    for image_path in wsi_paths:
        print('extract_patches_from_wsi: %s' % image_path)

        img_name,wsi_image, rgb_image, level_used= wsi_ops.read_wsi(image_path)
        assert wsi_image is not None, 'Failed to read Whole Slide Image %s.' % image_path

        augmented_image,bounding_boxes, rgb_contour, image_open = wsi_ops.find_roi_bbox(rgb_image)

        patch_extractor.extract_fixed_size_patches_from_WSI(augmented_image,img_name,
                                                            level_used, bounding_boxes,
                                                            image_open, patch_save_dir
                                                            )
        wsi_image.close()
        patch_index+=1


if __name__ == '__main__':

    def_level = 0
    PATCH_SIZE=32
    patch_size = 27
    STEP = 16
    WSI_PATH = "/home/adminofourkioti/PycharmProjects/-graph-at-net/ColonCancer"

    patch_save_dir = "ColonCancerPatches"

    #extract__patches_from_wsi(WSIOps(), PatchExtractor())
    patch_extractor=PatchExtractor()
    wsi_paths = glob.glob(os.path.join(WSI_PATH, '*.bmp'))

    wsi_paths.sort()
    patch_index = 0
    for root, dirs, image_paths in os.walk(WSI_PATH, topdown=False):

        for image_path in image_paths:
            mat_files = glob.glob(os.path.join(root, "*epithelial.mat"))

            assert len(mat_files)==1
            epithelial_file=scipy.io.loadmat(mat_files[0])

            label = 0 if (epithelial_file["detection"].size==0) else 1

            label_folder = os.path.join(patch_save_dir, str(label))
            if not os.path.exists(label_folder):
                os.makedirs(label_folder)
                print("Directory Created ")

            if image_path.endswith('.bmp'):
                print('extract_patches_from_wsi: %s' % image_path)

                img_name, wsi_image, rgb_image, level_used = WSIOps().read_wsi(os.path.join(root,image_path))

                assert wsi_image is not None, 'Failed to read Whole Slide Image %s.' % image_path

                patch_extractor.extract_fixed_location_patches(root,
                                    wsi_image, img_name,
                                    label_folder,
                                    patch_size,patch_index)
                patch_index += 1



