# mnist_distribute
TensorFlow Distribute MNIST
  
#####分布式简单原理
PS：用于管理参数，对梯度求平均，相当于监管者。  
Worker：用于batch的训练迭代，相当于工作者。
  
#####如何使用
1,新建一个dataset文件夹  
```python
mkdir dataset
```
2,本实验是采用一个ps一个worker，因此一定要打开两个终端。
```python
#ps
python tensorflow_mnist.py --dataset_dir=`pwd`/dataset --result_dir=`pwd`/result  --job_name=ps
#worker 0
python tensorflow_mnist.py --dataset_dir=`pwd`/dataset --result_dir=`pwd`/result
```
