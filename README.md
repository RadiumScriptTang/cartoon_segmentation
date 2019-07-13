# 基于迁移学习的动漫角色语义分割

# 1.前言
基于语义分割的场景分析是计算机视觉中的一个基本问题，其目的是为图像中的==每个像素指定一个类别标签==。

近年来，以FCN的提出为转折点，语义分割得到了快速的发展，随后==PSPnet（金字塔场景解析网络）==也被提出，较好地结局了FCN中存在的语境关系不匹配的问题。

数据集方面，大量优质的数据集不断涌现，典型的有MS COCO、Pascal Voc 2012、ADE20K_MIT等等，这些数据集不仅为我们提供了大量优质的训练样本，同时也有许多杰出的前辈为我们提供了各种网络在这些数据集的预训练模型。

迁移学习的思想也给了我们很大的启发。在数据极端匮乏的情况下，one-shot、zero-shot等思想大有用武之地。

迁移学习的方法使我们避免了为大量数据进行人工标记，相反，我们只需要对少量的样本进行标记，再基于一些预训练模型进行二次训练，就可以得到与在大量数据下训练的结果。

# 2.构造数据集

### (0).数据集格式
这一点必须首先说明。我们知道，语义分割的输入是一张图片，输出也是一张图片，但事实上==输入输出的维度可能并不完全一致==，输入往往是一张彩色的图，包含rgb（红黄蓝）三个通道，可以看做一个==高×宽×通道数==的张量，而输出往往可能是一个==高×宽==的二维矩阵，其中每一个元素对应输入像素的类别。

因此考虑训练数据，每一行数据都应该有两张图片，一张是原图，可以理解成features，一张是带有类别标记的图片。

keras_segmentation官方博客给出了一个数据集的示例。其结构如下
```
─dataset1
  ├─annotations_prepped_test
  ├─annotations_prepped_train
  ├─images_prepped_test
  └─images_prepped_train
```

该数据集大致分为两部分，训练集和测试集。训练集又分为==图片（images）集==和==注解（annotations）集==。下面我们重点关注一些这个注解集。

打开注解集发现，图片全都是黑的。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190712235135883.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1JhZGl1bVRhbmc=,size_16,color_FFFFFF,t_70)
事实上并非纯黑，当我们把亮度调高之后，就能看到大致的轮廓。
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019071223592089.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1JhZGl1bVRhbmc=,size_16,color_FFFFFF,t_70)

官方对此的解释是，注解图片应当是一个与原图大小相同的图片。
```python
import cv2
import numpy as np

ann_img = np.zeros((30,30,3)).astype('uint8')
ann_img[ 3 , 4 ] = 1 # this would set the label of pixel 3,4 as 1
ann_img[ 0 , 0 ] = 2 # this would set the label of pixel 0,0 as 2

cv2.imwrite( "ann_1.png" ,ann_img )
```
上述是官方给出的一段制作annotation的代码，这里假设我们有一个30×30的图片，要为其创建注解，我们把[0,0]处的元素标记为了类别2，而[3,4]处标记为类别1。==类别0认为是背景（background）==


由于时间、人力等问题，我们人工构造大量动漫角色的数据集==并不现实==。因此我们制作一个小规模的数据集。

制备方法如下：

### (1).准备原始图片
我们可以轻易地从网络上搜集到许多包含动漫人物的图片。在该数据集中我们共收集==40张==，其中==30张作为训练集==，==10张作为测试集==。（如下图）
![训练数据](https://img-blog.csdnimg.cn/20190712230206542.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1JhZGl1bVRhbmc=,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190712230231723.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1JhZGl1bVRhbmc=,size_16,color_FFFFFF,t_70)
### (2).标记数据
此处我们使用数据标记工具==labelme==，这款工具可以让我们方便地在图上对关键区域进行标记。

标记数据时间很苦的事情，俗称人体描边。（以下为示例）
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190713001141465.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1JhZGl1bVRhbmc=,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019071300115822.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1JhZGl1bVRhbmc=,size_16,color_FFFFFF,t_70)
尽管描边非常辛苦，但是看着这些卡哇伊的角色，也就不觉得累了（题外话）

标记完成后，labelme会帮助我们保存==json文件==，这个json文件记录了我们的标记点的顺序和位置。

### (3).导出为mask型png
labelme自带了命令
```
labelme_json_to_dataset <文件名>.json
```
这条命令可以帮助我们把json转成带有mask的png图片，但==缺点是一条命令只能转换一个文件==，对于批量文件，可以用python写脚本处理(一下是一个示例）。
```python
import os
for i in range(1,n):
    os.system("labelme_json_to_dataset " + "test/" + str(i) + ".json")
```

这样我们就得到了一个结果集。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190713004050807.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1JhZGl1bVRhbmc=,size_16,color_FFFFFF,t_70)
其中每个文件夹中包含了四个文件， ==.png文件==是我们需要的，我们把它设法提取到一个文件夹中。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190713004154369.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1JhZGl1bVRhbmc=,size_16,color_FFFFFF,t_70)
可以看到，这些png图片都是以黑色为背景，以rgb(128,0,0)为mask的图片。这离我们需要的注解格式还有最后一步距离。

### (4).注解化
最后一步，我们把上述png图片的所有红色mask区域的值都设置为1.


经过一顿猛如虎的操作，我们就得到了我们自创的数据集——动漫人物数据集，其结构同demo数据集一致,结构如下。
```
─dataset2
 ├─test_anno
 ├─test_images
 ├─train_anno
 └─train_images
```

# 3.训练模型
这里我们采用比较先进的PSPnet。在keras_segmentation的框架下，训练这样一个网络并不是难事。

```python
import keras_segmentation

model = keras_segmentation.models.pspnet.pspnet_101(n_classes=2)


model.train(
    train_images =  "dataset2/train_images/",
    train_annotations = "dataset2/train_anno/",
    checkpoints_path="tmp/pspnet_1",
    epochs=5
)
```
只需要指定好==所选的网络、训练的原图和注解文件==，并指定好==checkpoints的存放路径==，以及==训练的epochs==。

一方面，自己从底层搭建模型，可能会出错，这就会导致浪费大量的计算资源，另一方面，我水平也不够（根本原因），索性就直接用keras_segmentation提供的模型吧。

同样的，我们再基于预训练模型来train一个model，可以看做一种==迁移学习==
```python
import keras_segmentation

model = keras_segmentation.pretrained.pspnet_101_voc12()


model.train(
    train_images =  "dataset2/train_images/",
    train_annotations = "dataset2/train_anno/",
    checkpoints_path="tmp/pspnet_1",
    epochs=5
)
```

这里要吐槽一下官方文档，官方文档在描述迁移学习的时候，提到了一种transfer_weights方法，但事实上包里面压根没这么个方法。

根据官方的设计，==训练似乎只能在linux机上进行==，虽然官方没有明说，但当我在Windows上运行训练的时候，报出来一个莫须有的错误，使得我非常恼火。大致是说注解文件夹下没有响应的文件。但事实上我用os.path.exist()方法检查返回结果是True。天无绝人之路，我们碰巧有一台linux机，就是上次想象出来的那台，刚好能够派上用场。

需要特别注意的是这个==checkpoints==，这里真是太坑爹了，害我白跑了两个晚上。在keras_segmentation中，模型并不通过model.save()方法进行保存，虽然可以这样保存，但是再次读取之后会出现一些莫名其妙的问题。事实上模型的参数都是保存在checkpoints中，加载时可以通过keras_segmentation.predict.model_from_checkpoints(checkpoints_path)进行加载。至于为什么如此设计，咱也不知道，咱也不敢问。

# 4.模型评估
这里的准确率主要通过IoU（Intersection over Union）来反映，也就是预测结果与实际结果的交集。Keras_segmentation为我们提供了一个评估方法==keras_segmentation.predict.evaluate( model=None , inp_images=None , annotations=None , checkpoints_path=None )==。该方法接受一个模型、测试图片集和注解集，并输出测试集上的IoU。

但是当我们调用此方法的时候，竟然报了一行错误
```
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/gengshiqi/.local/lib/python3.6/site-packages/keras_segmentation/predict.py", line 111, in evaluate
    assert False , "not implemented "
AssertionError: not implemented 
```
气急败坏的我打开这个包的evaluate方法一看，差点没背过气来
```python
def evaluate( model=None , inp_inmges=None , annotations=None , checkpoints_path=None ):

        assert False , "not implemented "

        ious = []
        for inp , ann   in tqdm( zip( inp_images , annotations )):
                pr = predict(model , inp )
                gt = get_segmentation_arr( ann , model.n_classes ,  model.output_width , model.output_height  )
                gt = gt.argmax(-1)
                iou = metrics.get_iou( gt , pr , model.n_classes )
                ious.append( iou )
        ious = np.array( ious )
        print("Class wise IoU "  ,  np.mean(ious , axis=0 ))
        print("Total  IoU "  ,  np.mean(ious ))
```
方法的第一行竟然就是==assert False？？？==，还非常诚实地说道这个方法还没实现。我们把这行断言注释掉，还是运行不通。这肯定不是什么牛人写的包，但为了评估模型，evaluate又很难绕开，我只好免为其难把这个方法补全了,其实逻辑也不是很绕（以下是对该方法的修正）

```python
def evaluate( model=None , inp_images=None , annotations=None , checkpoints_path=None ):

	names = os.listdir(inp_images)
	images_annotations = [(os.path.join(inp_images,name),os.path.join(annotations,name)) for name in names]

	ious = []
	for inp , ann   in images_annotations:
		pr = predict(model , inp )
		gt = get_segmentation_arr( ann , model.n_classes ,  model.output_width , model.output_height  )
		gt = gt.argmax(-1)
		gt = gt.reshape(pr.shape)
		iou = metrics.get_iou( gt , pr , model.n_classes )
		ious.append( iou )
	ious = np.array( ious )
	print("Class wise IoU "  ,  np.mean(ious , axis=0 ))
	print("Total  IoU "  ,  np.mean(ious ))
```

对此，我已经修复了该方法，并发布了升级版的python包。该包目前还不能在pip上安装，朋友们可以从我的GitHub上下载，然后通过
```
python setup.py install
```
进行安装。GitHub地址为https://github.com/RadiumScriptTang/keras_segmentation

由于计算资源非常珍贵，我们没能尝试FCN等其他模型。我们不妨称呼psp在VOC2012数据集上的预训练模型叫做模型1，psp直接在动漫数据集上的训练模型称为模型2，预训练模型在动漫数据集的迁移学习模型称为模型3，实验结果如下
模型|模型1|模型2|模型3
--|--|--|--
背景IoU| 0.6729|0.6732|0.9083
动漫角色IoU|0| 0.6777|0.9112
平均IoU| 0.3365| 0.6754|0.9098

结果证明，迁移学习的成果还是相当明显的。

# 5.模型分析
我们可以借助keras 提供的方法对模型进行可视化，这样可以清晰得看到网络结构。

在PSPnet中，作者引入了金字塔池化模块，如下图。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190713193142481.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1JhZGl1bVRhbmc=,size_16,color_FFFFFF,t_70)
Hengshuang Zhao 等人分析了许多FCN网络结构的失败案例，他们观察到==语境关系不匹配==、==类别混淆==和==不明显的类别==等问题，并认为引入金字塔池化模块可以很好的结局这个问题。

在传统的psp网络中，金字塔池化层前的步骤基本也是普通的下采样操作。其结构如下。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190713194043307.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1JhZGl1bVRhbmc=,size_16,color_FFFFFF,t_70)
靠近底端的部分即为金字塔池化层。psp_101在上述基础上，在金字塔池化层前添加了许多交替的卷积层和batchNorm层。我推测大概有101层。

我们此处并不讨论why pspnet works（因为我也看不透）。

# 6.效果展示
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190713174055423.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1JhZGl1bVRhbmc=,size_16,color_FFFFFF,t_70)
（上图为直接训练模型的结果）
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190713173826771.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1JhZGl1bVRhbmc=,size_16,color_FFFFFF,t_70)
（上图为迁移学习模型的结果）

# 7.拓展延伸
基于上述工作，我们可以进一步的从动漫中抠出表情包。当然这其中涉及了一些视频分解、gif合成的问题，这里不做赘述。脚本我已经写好了，此处展示一下结果。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190713214956260.gif)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190713215012894.gif)
可以看到效果比较不错，图中的雪花是由于gif已经过压缩。

# 8.总结反思
总的来说，达到了我们较为理想的结果。但由于==数据集有限==，尽管已经通过迁移学习的方法达到了较为理想的IoU，但可以想象，如果能够有更大规模的数据集，我们一定可以得到更好的效果。

# 9.尾声
可以预见，雷帅未来从事CV方向的概率比较渺茫，这可能是雷帅在CV方向探索的巅峰。尽管如此，雷帅仍然从中学到了许多东西。

# 10.References
[1].https://github.com/divamgupta/image-segmentation-keras
[2].Jonathan Long, Evan Shelhamer ,Trevor Darrell, Fully Convolutional Networks for Semantic Segmentation, In CVPR,2015
[3].Hengshuang Zhao, Jianping Shi, Xiaojuan Qi,Xiaogang Wang, Jiaya Jia ,Pyramid Scene Parsing Network, in CVPR,2017
