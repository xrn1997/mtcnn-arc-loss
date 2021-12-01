基于[MTCNN](https://blog.csdn.net/qq_36782182/article/details/83624357) +Arc-SoftmaxLoss的人脸识别项目

第一步到第六步属于MTCNN的人脸检测阶段，第七步到第九步属于使用Insight Face 的人脸分类阶段。

注：看效果的话，可以直接执行第九步，项目中保存了训练过的网络，执行之前别忘了在相应的目录中按照正确的要求存上你的照片。

第一步：下载[CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) 数据集并解压图片

第二步：修改address.py中的CelebA存储路径。

第三步：运行data文件夹下的pro_gen_data.py，生成P、R、O网络的所需要的训练集。

第四步：修改address.py中的CelebA训练集路径。

第五步：运行trains文件夹下的pro_train.py 依次训练P、R、O网络，会生成param文件夹下的相关文件。

第六步：运行detect.py测试训练效果，该文件中有2段程序执行入口，其中一段被注释了。该步骤运行图片人脸检测那一段代码，注释掉另一段（视频人脸识别）。详情见代码注释。

第七步：运行face_train.py训练face网络，也会在param文件夹下生成相关的文件。

第八步：测试识别率，运行test.py测试网络的识别准确率，由于测试集、训练集、验证集取自同一个数据集，所以不可避免的有过拟合现象。

第九步：运行detect.py中的main（视频人脸识别），注释掉第六步执行的main（图片人脸检测）。该处将使用摄像头进行人脸识别，图片和相关代码请根据注释提前设置好。