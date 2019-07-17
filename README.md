# mnist_distribute
TensorFlow Distribute MNIST
      
## Distributed Simple Principle
PS：It is used to manage parameters and average the gradient, which is equivalent to the supervisor.      
Worker：The training iteration for batch is equivalent to the worker.    
  
## How to use?
1. mkdir /dataset, and put in the MNIST dataset.      
```python
mkdir dataset
```
2. This experiment uses one PS and one worker, so we must open two terminals. 
```python
#ps
python tensorflow_mnist.py --dataset_dir=`pwd`/dataset --result_dir=`pwd`/result  --job_name=ps
#worker 0
python tensorflow_mnist.py --dataset_dir=`pwd`/dataset --result_dir=`pwd`/result
```
