# GANs for sequences modeling
Supervised by Prof. [Jonathan Terhorst](http://www-personal.umich.edu/~jonth/).

This repository consists of work of writing an honor thesis. <br />The topic of this thesis is <br />
Below is the details of progress:<br />
-  We first try some toy examples by trainin a GANs to model a mixed exponential distribution. <br />

    The sequence looks like: 100\*exp(1), 1/100\*exp(1), 100\*exp(1), 1/100\*exp(1)<br />
    
    In the [3/17 notebook](https://github.com/Pengjp/gene_research/blob/master/3_17_Pre_explore.ipynb), I made a training dataset with 50000 rows and 1000 columns then fit into a vanilla GAN. After training, I made two plots to show the generated data. I separated odd and even columns and calculated even for each columns of the two sepatated data. So the length of x-axis of each plot is 500 (1000 / 2). We can observe that the generated data is getting close to the real distribution. However, this is not a good to way to subjectively judge the performance of this GAN. <br />
    Therefore,
