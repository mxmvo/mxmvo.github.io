---
layout: post
title: Geodesic Convolution
categories: [Thesis]
---

In this post shows an idea to generalise convolutional layers from images (functions on $\mathbb{R}^2$) to functions on triangulated manifolds. It will focus on the convolution of a kernel with a function. In a future post we will address the problem of input channels and output channels. The ideas presented here are based on [this paper](https://www.cv-foundation.org/openaccess/content_iccv_2015_workshops/w22/papers/Masci_Geodesic_Convolutional_Neural_ICCV_2015_paper.pdf).

---

Post Structure:
- Theory: introducing the idea and explaining the motivation
- The Kernel: The kernel used in the examples
- Example 1: The Sphere
- Example 2: The Body

---

## Theory

The idea behind standard convolutional layer in a neural network is to extract local information about the function/image. Typically this is done using weights from a kernel. This local information can become more and more abstract by adding multiple convolutional layers after one another. The way of extracting information is the same on the entire image, using this type of weight sharing greatly reduces the dimension of the weight space, and should make training a model simpler.

If the input is an image, then the network takes a local weighted average around a point. This operation is represented by the convolution operator.

$$
\begin{align*}
f\star g (x) = \sum_{n}f(n)g(x-n)
\end{align*}
$$

In the case of the image, the underlying structure is flat $2$ dimesnional euclidean space, and the pixels are spaced out evenly. In the case of a triangulated surface with a function, one cannot use this definition of the convolution, e.g. substraction is not defined on the triangulated manifold. 

Still the idea implemented here is the same, the layer should extract local information of the function around a points. Similar as in the article we explain the idea of an patch and then define the Geodesic Convolution. 

---

## Patch

Around each point we will have a neighborhood that is equivalent to a disk. On this disk we can define our kernel and then define the convolution by using this map from the neighborhood to the disk. 

More formally, define the set 

$$
\begin{align*}
 B_r(x):= \{ y \in M | d(x,y) < r \}
\end{align*}
$$ 

the set of all point such that the geodesic distance is smaller than $r$. Additionally we can compute the actual geodesic between the points $y$ and $x$. These geodesics have a representative tangent vector in the tangent space $T_xM$. After choosing a basis of the tangent space we can calculate the angles of the vectors denote this angle as $\Theta_x(y)$. This way we can make a function 

$$
\begin{align*}
\phi_x & : B_r(x) \to D_r\\
y & \mapsto (d(x,y), \Theta_x(y))
\end{align*}
$$

Note that this definition of a patch only makes sense if $r$ is small enough, such that $B_r(x)$ is actually topological equivalent to a disk. This makes sure that the cut locus is not represented in the disk, meaning the function is continuous.  

Then the geodesic convolutional operator is defined as $f\star g = \sum_{\rho, \theta} f(\phi_x^{-1}(\rho,\theta))k(\rho, \theta)$

---

### Tangent Vectors

Given a triangulated surface, how does one speak of a tangent space at a point, this point where triangles come together, will not have a natural tangent space. We overcome this issue, by mapping the neighborhood to $\mathbb{R}^2$ by keeping the ratio of triangle angles the same.  

---

### Hardbinning

The function is only defined on the vertices of the manifold, then $f(\phi_x^{-1}(\rho,\theta))$ might not always be well defined. Computationally we bin each point in $B_r(x)$ into an angular bin and a radial bin. Then the kernel is a matrix with weights for each bin. Then we take the local average, if the mesh does not have a representative point for a bin then we the value is assumed to be zero. 

---

### Softbinning

As a soft binning principle, one can smooth out the mapped points on the disk, this will make them contribute to bins that potentialy didn't have a member. This smoothing out principle will potentially reduces the importance of the triangulation. Althoug smoothing out too much might decrease the power of the convolution in later stadia of the network. 

---

### Bin normalisation

If the vertices are not spread out nice and evenly, taking an unweighted sum might heavely favor a bin if it contains a lot of vertices. Or similarly, a vertex in an area that is sparsely populated will not obtain as big a value as a point which is densely populated. Hence we use bin normalisation. Where each bin has at most weight $1$ evenly spread out among its members. simmilarly on a path we weight the num by the total number of vertices. 

---

# Notes

This method has a drawback, namely we have to choose a basis. Therefor the orientation of the patch is not consistant. The reason why we have to choose a basis is because there is not a natural way of making a mapping between two points.
This is something the image convolution did have. Furthermore, with image convolutions one can set stepsizes to reduce the height and width of an image, this does not have an equivalent in the triangulated case.

What can also be regarded as a drawback is the creation of the patches, we assume the triangulated data is behaving. This might limit the generalisation. 

---

### Code


```python
import sys
import vtki
import torch
import scipy
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt

from plyfile import PlyData, PlyElement
from tqdm import tqdm
from vtki import PolyData
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D

# Home made modules
from modules.trimesh import trimesh
from modules.fast_marching_method import FMM
from modules.gradient_walk.linear_walk import LinearWalk
from modules.adj_matrix import adjacency_matrix_fmm
from modules.geometry_functions import laplace_beltrami_eigenfunctions

%matplotlib inline
%load_ext autoreload
%autoreload 2
```

A function to load the mesh


```python
def trimesh_from_ply(f_name):
    data = PlyData.read(f_name)
    
    data_vert = np.vstack([list(vertex) for vertex in data['vertex'].data])
    data_tri = np.vstack(data['face'].data['vertex_indices'])
    return trimesh(data_vert,data_tri)

# Bin parameter
p_bins = 5
t_bins = 16
```

---

# The kernel

Since this method chooses an orientation of the kernel randomly, we take a kernel that is rotation invariant. The kernel is the gaussian kernel.

Below we construct the kernel

---


```python
# Some small functions
sigma_p = 1
p_exp = lambda x: np.exp(-x**2/(2*sigma_p**2))
p_trans = lambda r: r/p_bins
t_trans = lambda t: 2*np.pi*t/t_bins

def plot_disk(t,r,z):
    fig = plt.figure()

    plt.subplot(projection="polar")
    plt.pcolormesh(t, r, z)
    plt.grid()

    plt.show()
```


```python
# Make the kernel and plot it
rad = np.linspace(0, 5, p_bins)
azm = np.linspace(0, 2 * np.pi, t_bins)
r, th = np.meshgrid(rad, azm)

def make_kernel(p_bins,t_bins):
    kernel = np.zeros((t_bins, p_bins))

    for i in range(p_bins):
        r_1 = p_trans(i)
        val = p_exp(r_1)
        for j in range(t_bins):
            kernel[j,i] = val
    return kernel/kernel.sum()

kernel = make_kernel(p_bins, t_bins)
plot_disk(th, r , kernel)
```

![png](/assets/images/geodesic_convolution/output_6_1.png)


---

# Example 1: The Sphere

In this example we take the isohedral sphere.


```python
# open the mesh
file_sphere = '../test_data/sphere_iso_6c.ply'
sphere = trimesh_from_ply(file_sphere)
```

The below function creates the adjacency matrix:
The innerworking can be described as 
- for each vertex do
    - find the points in a certain geodesic radius
    - find for each of these points the geodesic
    - bin each point in a angular and radial bin
- return the matrix

Right now the matrix has dimension NxNxPxT, where N are the number of vertices P the number of radial bins and T the number of angular bins. 


```python
# Make the adjacency matrix
adj_sphere = adjacency_matrix_fmm(sphere, p_max = .5, p_bins = p_bins, t_bins = t_bins)
```

    100%|██████████| 642/642 [03:10<00:00,  3.49it/s]


## Sparse Matrices

The creation of the adjacency matrix can be done as a preprocessing step. Now the adjacency matrix can become quite big, while a lot of entries are actually zero. Thus it makes sense to use sparse matrices, in this data structure one only remembers the rowpointers, columns and the data point. This saves significant memory if the matrix is sparse, also certain calculations can be done more efficiently. 

Thus since at a later stage the adjacency matrix is represented as a sparse matrix, we cast the matrix to a sparse matrix. A sparse matrix has to have a 2 dimensional stucture, hence the reshape.

---

Below some extra fuctions that help handle the sparse matrices.

---


```python
def bin_normalisation(sp_mat):
    '''
    Normalize devide the vertex by the total number of vertices in the bin.
    Then also devide by the total number of points in the patch.
    '''
    ind = sp_mat.indices
    row = sp_mat.indptr
    
    res1 = []
    res2 = []
    for r in range(len(row)-1):
        col = ind[row[r]:row[r+1]]
        bins = col % (p_bins*t_bins)
        
        num_bins = len(set(bins))
        count_bin = Counter(bins)
        for c in bins:
            res1.append(count_bin[c])
            res2.append(num_bins)
    return scipy.sparse.csr_matrix((1/(np.array(res1)*np.array(res2)),ind, row)) 

def sparse_reshape(sp_mat):
    '''
    The sparse matrix is 2 dimensional, given the number of bins
    we can find which vertex belongs where
    '''
    ind = sp_mat.indices
    row = sp_mat.indptr
    
    # Find which bins the points belong to. 
    sp_t = ind % t_bins # angle
    sp_p = ( ind % (t_bins*p_bins))//t_bins # radius
    sp_c = ind // (t_bins*p_bins) #column
    diff = row[1:] - row[:-1] 
    sp_r = [i for i in range(len(diff)) for _ in range(diff[i])] #row
    
    return sp_r, sp_c, sp_p, sp_t
    
def make_conv_matrix(sp_mat, kernel):
    '''
    Combine the adjacency matrix with the kernel to make the convolution matrix.
    '''
    # Read lists from the sparse matrix
    num_vert = sp_mat.shape[0]
    
    sp_r, sp_c, sp_p, sp_t = sparse_reshape(sp_mat)
    data = sp_mat.data.astype(np.float)
    
    # Find the weights of each bin and multiply by the amount a vertex should contribute. 
    sp_d = []
    indptr = sp_mat.indptr
    for i in range(len(indptr)-1):
        sl = slice(indptr[i],indptr[i+1])
        rel = kernel[sp_t[sl],sp_p[sl]]*data[sl]
        sp_d.extend(rel/rel.sum())


    # Cast this again to a sparse matrix
    conv_mat = scipy.sparse.coo_matrix((sp_d, (sp_r,sp_c)), shape = (num_vert,num_vert))
        
    return conv_mat
    
```


```python
# Make a sparse matrix out of the adjacenct matrix. 
# And normalize it.
sparse_sphere = scipy.sparse.csr_matrix(adj_sphere.reshape(len(sphere.vertices),-1))
sparse_sphere = bin_normalisation(sparse_sphere)

# Make the convolution matrix
conv_sphere = make_conv_matrix(sparse_sphere, kernel)
```


```python
# Our function has value 1 on a vertex and is zero otherwise.
func = np.zeros(len(sphere.vertices))
func[0] = 1

# The convoluted function
conv_func = conv_sphere.dot(func)
```


```python
# Plotting the original and the convoluted function. 
plotter = vtki.Plotter(shape=(1, 2))

color_map = 'jet'

plotter.subplot(0,0)
polydata = PolyData(sphere.vertices, np.c_[[[3]]*len(sphere.triangles),sphere.triangles])
plotter.add_text('Initial function', position=None, font_size=20)
plotter.add_mesh(polydata, scalars = func, cmap = color_map)
plotter.view_vector([0,0,-1], viewup = [1,1,0])


plotter.subplot(0, 1)
polydata = PolyData(sphere.vertices, np.c_[[[3]]*len(sphere.triangles),sphere.triangles])
plotter.add_text('Convoluted function', position=None, font_size=20)
plotter.add_mesh(polydata, scalars = conv_func, cmap = color_map)
plotter.view_vector([0,0,-1], viewup = [1,1,0])
                 
plotter.show()
```


![png](/assets/images/geodesic_convolution/output_16_0.png)


# Example 2: The body

This example show a body traingulation of from the [FAUST](http://faust.is.tue.mpg.de) dataset. This triangulation is a bit more irregular than the sphere. 

We have already precomputed the adjacency matrix (hardbinning), to calculate this would take roughly one hour. To make the adjacency matrix we have used the following settings:
- `p_max = 0.03`
- `t_bins = 16`
- `p_bins = 5`

The function on which we will take the convolution is the an eigenvector of the laplace beltrami operator with added noise. The kernel will again be the gaussian kernel, ideally this kernel should be able to help with the noise that we added. 


```python
file_adj = '../test_data/tr_reg_004_raw.npz'
file_body = '../test_data/tr_reg_004.ply'

p_bins = 5
t_bins = 16
adj_mat = scipy.sparse.load_npz(file_adj)
adj_body = bin_normalisation(adj_mat)
body = trimesh_from_ply(file_body)
```

---

Making a new kernel with the new parameters, we will take a more interesting function as well. 

---


```python
# get a laplace function
val, vec = laplace_beltrami_eigenfunctions(body,k = 50)

# To make sure we get the same result
np.random.seed(0)
laplace_func_orig = vec[:,20]
laplace_func = laplace_func_orig + np.random.normal(scale = .15, size = len(laplace_func_orig))
```


```python
# Plotting the original and the convoluted function. 
plotter = vtki.Plotter(shape=(1, 2))

color_map = 'jet'

plotter.subplot(0,0)
polydata = PolyData(body.vertices, np.c_[[[3]]*len(body.triangles),body.triangles])
plotter.add_text('Original function', position=None, font_size=20)
plotter.add_mesh(polydata, scalars = laplace_func_orig, cmap = color_map)
plotter.view_vector([0,0,1], viewup = [0,1,0])


plotter.subplot(0, 1)
polydata = PolyData(body.vertices, np.c_[[[3]]*len(body.triangles),body.triangles])
plotter.add_text('Noisy function', position=None, font_size=20)
plotter.add_mesh(polydata, scalars = laplace_func, cmap = color_map)
plotter.view_vector([0,0,1], viewup = [0,1,0])
                 
plotter.show()
```


![png](/assets/images/geodesic_convolution/output_21_0.png)



```python
kernel = make_kernel(p_bins, t_bins)
conv_body = make_conv_matrix(adj_body, kernel)

```


```python
# The convoluted function
conv_func = conv_body.dot(laplace_func)
```

## Softbinning

For the soft binning we smooth out the projected point on the disk, so that is also contributes to other bins. This is also done using a gaussian blur. 

Because we use a gaussian blur to smooth out points, and later use a gaussian filter, we do not expect to see a huge difference. Given a different filter, one might expect to see beter results. 


```python
sigma_p = 1

rad = list(range(p_bins))
azm = list(range(t_bins))
r_o, th_o = np.meshgrid(rad, azm)
r, th = p_trans(r_o), t_trans(th_o)
y1 = (r*np.cos(th)).reshape(-1)
y2 = (r*np.sin(th)).reshape(-1)
y = np.stack([y1,y2])


def radial_smoothing(r_0, t_0, y = y):
    r_0, t_0 = p_trans(r_0), t_trans(t_0)
    x = np.array([[r_0*np.cos(t_0)], [r_0*np.sin(t_0)]])
    
    z = p_exp(np.linalg.norm(y-x, axis = 0)).reshape(t_bins, p_bins)
    return z/z.sum()
```


```python
def soft_bins(adj_body, p_bins = p_bins, t_bins = t_bins):
    sp_r, sp_c, sp_p, sp_t = sparse_reshape(adj_body)
    
    num_vert = adj_body.shape[0]
    data = adj_body.data
    
    ind_range = np.array([j for j in range(t_bins*p_bins)])
    new_ptr, new_i, new_data = [0], [], []
    prev_row = 0
    for i in tqdm(range(len(sp_r))):
        if sp_r[i] != prev_row:
            prev_row = sp_r[i]
            ptr = i*t_bins*p_bins
            new_ptr.append(ptr)
        
        new_data.extend(list(data[i]*radial_smoothing(sp_p[i],sp_t[i]).T.reshape(-1)))
        new_i.extend(list(sp_c[i]*t_bins*p_bins+ ind_range))
    new_ptr.append(len(new_data)) 
    return scipy.sparse.csr_matrix((new_data, new_i, new_ptr))
        
        
        
        
```


```python
sparse_body = soft_bins(adj_body)
conv_smooth = make_conv_matrix(sparse_body, kernel)
```

    100%|██████████| 230596/230596 [00:28<00:00, 8159.45it/s]



```python
# The convoluted function
conv_func_smooth = conv_smooth.dot(laplace_func)
```

---

If we plot the convoluted functions, there is not much difference between the hardbinned and the softbinned. This might be because the triangulation around a point is fine grained enough. But also because the kernel is a gaussian blur, this will hide the effect of the smoothing out of a point. 

---


```python
# Plotting the original and the convoluted function. 

plotter = vtki.Plotter(shape=(2, 2))

color_map = 'jet'

plotter.subplot(0,0)
polydata = PolyData(body.vertices, np.c_[[[3]]*len(body.triangles),body.triangles])
plotter.add_text('Original function', position=None, font_size=20)
plotter.add_mesh(polydata, scalars = laplace_func_orig, cmap = color_map)
plotter.view_vector([0,0,1], viewup = [0,1,0])


plotter.subplot(0, 1)
polydata = PolyData(body.vertices, np.c_[[[3]]*len(body.triangles),body.triangles])
plotter.add_text('Noisy function', position=None, font_size=20)
plotter.add_mesh(polydata, scalars = laplace_func, cmap = color_map)
plotter.view_vector([0,0,1], viewup = [0,1,0])
                 
plotter.subplot(1,0)
polydata = PolyData(body.vertices, np.c_[[[3]]*len(body.triangles),body.triangles])
plotter.add_text('Hard bin', position=None, font_size=20)
plotter.add_mesh(polydata, scalars = conv_func, cmap = color_map)
plotter.view_vector([0,0,1], viewup = [0,1,0])


plotter.subplot(1, 1)
polydata = PolyData(body.vertices, np.c_[[[3]]*len(body.triangles),body.triangles])
plotter.add_text('Soft bin', position=None, font_size=20)
plotter.add_mesh(polydata, scalars = conv_func_smooth, cmap = color_map)
plotter.view_vector([0,0,1], viewup = [0,1,0])
                 
plotter.show()
```


![png](/assets/images/geodesic_convolution/output_30_0.png)

