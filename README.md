# GANs for sequences modeling
Supervised by Prof. [Jonathan Terhorst](http://www-personal.umich.edu/~jonth/).

This repository consists of work of writing an honor thesis. The topic of this thesis is 
Below is the details of progress:
-  We first try some toy examples by trainin a GANs to model a mixed exponential distribution. 
    The sequence looks like: 100\*exp(1), 1/100\*exp(1), 100\*exp(1), 1/100\*exp(1)
    In the [3/17 notebook](https://github.com/Pengjp/gene_research/blob/master/3_17_Pre_explore.ipynb), I made a training dataset with 50000 rows and 1000 columns then fit into a vanilla GAN. After training, I made two plots to show the generated data. I separated odd and even columns and calculated even for each columns of the two sepatated data. So the length of x-axis of each plot is 500 (1000 / 2). We can observe that the generated data is getting close to the real distribution. However, this is not a good to way to subjectively.
- Then I took a step back to train a GAN for modeling an exponential distribution. <br />
  The sequence consists of numbers sampling from an exponentail distribution with rate 1. Then I scale the sequence with 100. This is sequence is one of the peak in the mixed exponential distribution I tried to model before. The result is in [3/30 notebook ](https://github.com/Pengjp/gene_research/blob/master/3_30_single_expo_model.ipynb). <br />  The first plot is a density plot showing the target distribution after transformation. The reason to do transformation is that deep learning perfers the input in a small range like we always do rescaling for image data. For this task, I added few layers to the original GAN and lower the learning rate for both generator and discirminator by referencing many tips of training GANs.
