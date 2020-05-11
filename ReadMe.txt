Code deveploded by Mr Zhongjie Xu, Msc Student, Qmul

Requirements:

      1. Python 3.x
      2. PyTorch 1.5
      3. torchvision 0.5.0


Train：

You have even two data sets to choose one from Cifr10 (another) and the other from Mnist. For Googlenet, Resnet and VGG, these two networks need to comment out Cifr10 in train.py, and uncomment the code in the Mnist part. At the same time, they need to modify the shape in Model.py. For Senet, you can directly run Cifr10.py and mnist.py.
