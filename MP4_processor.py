import cv2
import os
import imageio
import keras_segmentation

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
        cv2.imwrite(os.path.join(outp,"seg/seg_"+name),img_origin)

def mp4Processor(file_path,model):
    mp4 = cv2.VideoCapture(file_path)
    assert mp4.isOpened(), "Failed to open mp4 file"
    rval,frame = mp4.read()
    i = 1
    # 分解视频图片
    while rval:
        cv2.imwrite("mp4_images/" + str(i) + ".png",frame)
        i += 1
        rval,frame = mp4.read()

    # 抠图
    output_segmentations(model,"mp4_images/","gif_images/")

    seg_names = os.listdir("gif_images/seg/")
    frames = [imageio.imread(os.path.join("gif_images/seg",name)) for name in seg_names]
    imageio.mimsave("output.gif",frames,'GIF',duration=0.05)

if __name__ == "__main__":
    model = keras_segmentation.predict.model_from_checkpoint_path("tmp/pspnet_transfer")
    mp4Processor("1.mp4",model)




