import keras_segmentation
import cv2
import os

def output_segmentations(model,inp=None,outp=None):
    assert inp and outp,"None type inp and outp is not allowed"
    names = os.listdir(inp)
    for name in names:
        model.predict_segmentation(inp=os.path.join(inp,name),out_fname=os.path.join(outp,name))
        img = cv2.imread(os.path.join(outp,name))
        img_origin = cv2.imread(os.path.join(inp,name))
        h, w, c = img.shape
        for i in range(h):
            for j in range(w):
                if img[i, j, 0] == 197:
                    img_origin[i, j] = 255
        cv2.imwrite(os.path.join(outp,"seg_"+name),img_origin)

if __name__ == "__main__":
    model = keras_segmentation.predict.model_from_checkpoint_path("tmp/pspnet_1")
    output_segmentations(model,"dataset2/test_images","outputs")