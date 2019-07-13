import keras_segmentation

model = keras_segmentation.pretrained.pspnet_101_voc12()


model.train(
    train_images =  "dataset2/train_images/",
    train_annotations = "dataset2/train_anno/",
    checkpoints_path="tmp/pspnet_1",
    epochs=5
)



