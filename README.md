# dummy
[TOC]
# 数据预处理过程总结
## 目标：CT和MR的预处理、CT图像去除枕线(可选)、CT和MR的对齐与配准、后期数据增强
## 方法简介：
> 首先根据Deep Learning with Magnetic Resonance and Computed Tomography Images我们明确了对MR和CT数据的基本处理流程，具体内容可以见上文的解析(位于相关内容学习文件夹里)

目前总结出的数据处理流程一共有三种方法：
1) 使用spm中的clinical tools进行处理
2) 使用吴老师的方法用ITKSNAP配准
3) 使用simpleITK分步骤的处理

## 具体描述
首先我们是用的实例数据一共有三例

| patient ID | CT扫描     | MR扫描     |      |
| :--------- | ---------- | ---------- | ---- |
| 0000069963 | 512×512×27 | 512×512×18 |      |
| 0008519933 | 512×512×69 | 512×512×18 |      |
| 0017041336 | 512×512×86 | 512×512×18 |      |

### 方法一
&nbsp;&nbsp;&nbsp;&nbsp;本方法使用spm8中的一个临床工具包([Clinical Toolbox](https://www.nitrc.org/docman/?group_id=881))提供的batch来进行MR和CT的预处理，包括CT的重采样、配准、标准化；MR的重采样、配准、偏场校正、标准化。**具体操作如下**

**1、转化dcm文件为nii文件(本步骤为通用步骤，在以下两种方法中不再赘述)**
<br>dcm文件到nii文件的转化有两种方法：

* 1 使用simpleITK包读取保存
* 2 使用dcm2niix(注意本次数据集为压缩的dcm文件，不能直接使用dcm2nii转化，而要改用dcm2niix)

**2、使用spm配准CT**
* 打开matlab，输入spm-fmri启动spm
* 打开batch 选择我们要使用的Clinic Toolbox中的CT normalization
* 添加对应的nii文件后，进行转化
* 转化结果直接保存在对应文件夹中(以w打头)

**3、使用spm配准MR(基本类似第二步)**
* 打开matlab，输入spm-fmri启动spm
* 打开batch 选择我们要使用的Clinic Toolbox中的MR normalization
  <img src="https://s2.ax1x.com/2019/04/11/AHadKI.png" width=256 height=256/>
  <img src="https://s2.ax1x.com/2019/04/11/AHaDVf.png" width=256 height=256/>
* 添加对应的nii文件后，进行转化



<img src="https://s2.ax1x.com/2019/04/11/AHawrt.png" width=256 height=256/>

* 转化结果直接保存在对应文件夹中(以w打头)

**4、结果记录**
以8519933号病人为例：
* CT数据及其information
<figure class="third">
    <img src="https://s2.ax1x.com/2019/04/11/AHaUxA.png" width=256 height=256>
    <img src="https://s2.ax1x.com/2019/04/11/AHara8.gif" width=256 height=256>
</figure>
* MR数据及其imformation
<figure class="third">
    <img src="https://s2.ax1x.com/2019/04/11/AHa0qP.png" width=256 height=256>
    <img src="https://s2.ax1x.com/2019/04/11/AHasIS.gif" width=256 height=256>
</figure>

### 方法二
使用itksnap

```
# Convert CT with force stack option
'/home/zhangtw/dcm2niix' -m y '/data/zhangtw/data1/0000069963/20180506/CT03322022'

# Convert MRI
'/home/zhangtw/dcm2niix' '/data/zhangtw/data1/0000069963/20180508/MR00924205'

# Affine registration
'/home/zhangtw/itksnap-3.8.0-beta-20181028-Linux-gcc64/bin/greedy' -a -dof 6 -d 3 -m NMI -threads 20 -i '/data/zhangtw/data1/0000069963/20180508/MR00924205/MR00924205_T2W_FLAIR_SENSE_20180508195024_601.nii' '/data/zhangtw/data1/0000069963/20180506/CT03322022/CT03322022_HX_NeuroVPCT_70kV_CTA_20180506114626_2.nii' -o '/data/zhangtw/CT_reg.txt'

# Convert the affine file from c3d format to ITK format
'/home/zhangtw/itksnap-3.8.0-beta-20181028-Linux-gcc64/bin/c3d_affine_tool' '/data/zhangtw/CT_reg.txt' -oitk '/data/zhangtw/CT_reg_itk.txt'

# Apply the affine transformation
'/home/zhangtw/itksnap-3.8.0-beta-20181028-Linux-gcc64/bin/c3d' '/data/zhangtw/data1/0000069963/20180508/MR00924205/MR00924205_T2W_FLAIR_SENSE_20180508195024_601.nii' '/data/zhangtw/data1/0000069963/20180506/CT03322022/CT03322022_HX_NeuroVPCT_70kV_CTA_20180506114626_2.nii' -reslice-itk '/data/zhangtw/CT_reg_itk.txt' -o '/data/zhangtw/CT_reg.nii.gz'

```
该方法生成CT_reg.nii.gz文件是将原始CT图像对准到MR图像后的结果，**转换后的CT文件size会变成MR图像的size，即512 × 512 × 18** 需要之后做进一步的resample和normalization
<br><br>结果图示：

<img src="https://s2.ax1x.com/2019/04/12/AbehjJ.gif" >




### 方法三
&nbsp;&nbsp;&nbsp;&nbsp;本方法使用simpleITK包中的方法分步骤进行数据预处理，**具体操作如下**

**1、导入模块**

```python
from __future__ import print_function
import importlib
from distutils.version import LooseVersion
import SimpleITK as sitk

# check that all packages are installed (see requirements.txt file)
required_packages = {'jupyter', 
                     'numpy',
                     'matplotlib',
                     'ipywidgets',
                     'scipy',
                     'pandas',
                     'SimpleITK'
                    }

problem_packages = list()
# Iterate over the required packages: If the package is not installed
# ignore the exception. 
for package in required_packages:
    try:
        p = importlib.import_module(package)        
    except ImportError:
        problem_packages.append(package)
    
if len(problem_packages) is 0:
    print('All is well.')
else:
    print('The following packages are required but not installed: ' \
          + ', '.join(problem_packages))
print(sitk.Version())
```
**2、读取数据**

```python
path_mr = '/data/zhangtw/MR_nii_test_data/test_MR_FLAIR17041336.nii.gz'
path_ct = '/data/zhangtw/CT_nii_test_data/test_CT_17041336.nii.gz'
image_mr = sitk.ReadImage(path_mr, sitk.sitkFloat32)
image_ct = sitk.ReadImage(path_ct, sitk.sitkFloat32)

In: print(image_mr.GetSize())
Out: (512, 512, 18)
print(image_mr.GetSpacing())
Out: (0.44921875, 0.44921875, 7.289997100830078)
```

**3、resample**

```python
# code resourse: https://www.atyun.com/23342.html
def resample_img(itk_image, out_spacing=[2.0, 2.0, 2.0], is_label=False):
    
    # Resample images to 2mm spacing with SimpleITK
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)

# Assume to have some sitk image (itk_image) and label (itk_label)
resampled_img_mr = resample_img(image_mr, out_spacing=[1.0, 1.0, 1.0], is_label=False)
resampled_img_ct = resample_img(image_ct, out_spacing=[1.0, 1.0, 1.0], is_label=False)
```
resample后的数据信息：
<img src="https://s2.ax1x.com/2019/04/11/AHdx6s.png" >
可以看到CT图像resample后的size变成了257 × 257 × 172，而MR图像resample后变成了230 × 230 × 131

**4、MR图像做N4 bias field correction**

```
#code resourse: https://github.com/bigbigbean/N4BiasFieldCorrection/blob/master/N4BiasFieldCorrection.py
maskImage = sitk.OtsuThreshold(resampled_img_mr, 0, 1, 200 )
inputImage = sitk.Cast(resampled_img_mr, sitk.sitkFloat32)
corrector = sitk.N4BiasFieldCorrectionImageFilter();
output_mr = corrector.Execute(inputImage, maskImage)
#sitk.WriteImage(output,"/data/zhangtw/n4.nii")
print("Finished N4 Bias Field Correction.....")
```

**5、图像配准**
* 定义可视化函数

```python
import matplotlib.pyplot as plt
%matplotlib inline

from ipywidgets import interact, fixed
from IPython.display import clear_output

# Callback invoked by the interact IPython method for scrolling through the image stacks of
# the two images (moving and fixed).
def display_images(fixed_image_z, moving_image_z, fixed_npa, moving_npa):
    # Create a figure with two subplots and the specified size.
    plt.subplots(1,2,figsize=(10,8))
    
    # Draw the fixed image in the first subplot.
    plt.subplot(1,2,1)
    plt.imshow(fixed_npa[fixed_image_z,:,:],cmap=plt.cm.Greys_r);
    plt.title('fixed image')
    plt.axis('off')
    
    # Draw the moving image in the second subplot.
    plt.subplot(1,2,2)
    plt.imshow(moving_npa[moving_image_z,:,:],cmap=plt.cm.Greys_r);
    plt.title('moving image')
    plt.axis('off')
    
    plt.show()

# Callback invoked by the IPython interact method for scrolling and modifying the alpha blending
# of an image stack of two images that occupy the same physical space. 
def display_images_with_alpha(image_z, alpha, fixed, moving):
    img = (1.0 - alpha)*fixed[:,:,image_z] + alpha*moving[:,:,image_z] 
    plt.imshow(sitk.GetArrayViewFromImage(img),cmap=plt.cm.Greys_r);
    plt.axis('off')
    plt.show()
    
# Callback invoked when the StartEvent happens, sets up our new data.
def start_plot():
    global metric_values, multires_iterations
    
    metric_values = []
    multires_iterations = []

# Callback invoked when the EndEvent happens, do cleanup of data and figure.
def end_plot():
    global metric_values, multires_iterations
    
    del metric_values
    del multires_iterations
    # Close figure, we don't want to get a duplicate of the plot latter on.
    plt.close()

# Callback invoked when the IterationEvent happens, update our data and display new figure.    
def plot_values(registration_method):
    global metric_values, multires_iterations
    
    metric_values.append(registration_method.GetMetricValue())                                       
    # Clear the output area (wait=True, to reduce flickering), and plot current data
    clear_output(wait=True)
    # Plot the similarity metric values
    plt.plot(metric_values, 'r')
    plt.plot(multires_iterations, [metric_values[index] for index in multires_iterations], 'b*')
    plt.xlabel('Iteration Number',fontsize=12)
    plt.ylabel('Metric Value',fontsize=12)
    plt.show()
    
# Callback invoked when the sitkMultiResolutionIterationEvent happens, update the index into the 
# metric_values list. 
def update_multires_iterations():
    global metric_values, multires_iterations
    multires_iterations.append(len(metric_values))     
```
* 先将原始图像可视化(**注意变量定义，本方法是将moving_image对齐到fixed_image**)：

```
moving_image = resampled_img_ct
fixed_image = output_mr
interact(display_images, fixed_image_z=(0,fixed_image.GetSize()[2]-1), moving_image_z=(0,moving_image.GetSize()[2]-1), fixed_npa = fixed(sitk.GetArrayViewFromImage(fixed_image)), moving_npa=fixed(sitk.GetArrayViewFromImage(moving_image)));
```

<img src="https://s2.ax1x.com/2019/04/11/AHBKJO.gif" >

* 进行初始对齐：

```
initial_transform = sitk.CenteredTransformInitializer(fixed_image, 
                                                      moving_image, 
                                                      sitk.Euler3DTransform(), 
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY)

moving_resampled = sitk.Resample(moving_image, fixed_image, initial_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

interact(display_images_with_alpha, image_z=(0,fixed_image.GetSize()[2]), alpha=(0.0,1.0,0.05), fixed = fixed(fixed_image), moving=fixed(moving_resampled));
```

<img src="https://s2.ax1x.com/2019/04/11/AHBuFK.gif" >

* 进行配准：

```
registration_method = sitk.ImageRegistrationMethod()

# Similarity metric settings.
registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
registration_method.SetMetricSamplingPercentage(0.01)

registration_method.SetInterpolator(sitk.sitkLinear)

# Optimizer settings.
registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
registration_method.SetOptimizerScalesFromPhysicalShift()

# Setup for the multi-resolution framework.            
registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

# Don't optimize in-place, we would possibly like to run this cell multiple times.
registration_method.SetInitialTransform(initial_transform, inPlace=False)

# Connect all of the observers so that we can perform plotting during registration.
registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations) 
registration_method.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration_method))

final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), 
                                              sitk.Cast(moving_image, sitk.sitkFloat32))
```

* 配准结果可视化：

```
moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

interact(display_images_with_alpha, image_z=(0,fixed_image.GetSize()[2]), alpha=(0.0,1.0,0.05), fixed = fixed(fixed_image), moving=fixed(moving_resampled));
```

<img src="https://s2.ax1x.com/2019/04/11/AHBCJU.gif" >

* 配准结果保存

```
sitk.WriteImage(moving_resampled, '/data/zhangtw/final_ct.nii.gz')
sitk.WriteImage(fixed_image, '/data/zhangtw/final_mr.nii.gz')
```

* 处理后文件的information
    * CT
    <figure class="third">
        <img src="https://s2.ax1x.com/2019/04/11/AHDLbn.png" width=256 height=256>
        <img src="https://s2.ax1x.com/2019/04/11/AHDv5V.gif" width=256 height=256>
    </figure>
    * MR
    <figure class="third">
    <img src="https://s2.ax1x.com/2019/04/11/AHDXEq.png" width=256 height=256>
    <img src="https://s2.ax1x.com/2019/04/12/AbuE0s.gif" width=256 height=256>
    </figure>   
