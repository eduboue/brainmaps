# Brainmaps Pipeline

## Overview
brainmaps is a set of yools for brain-wide mapping, preprocessing, and analysis of volumetric imaging data. 

The package is a set of classes and methods meant to analysze tERK/pERK imaging data. The file essentially has a class `Experiment` that holds a single experiment, and then another class `AnalysisSuite` which holds many experiments. So, let's say you have a set of 10 images, you can create an `AnalysisSuite` instance which has 10 `Experiment` instanes, one for each brain. There are sets of constuctors to build the `AnalysisSute` class, including one to build from config and another to build from a directory. 

Lastly, the package has a set of permutation functions to prepare your data and run a permutation. Then, there is a `PermutationResults` dataclass to hold your results. 


## Environment Setup

Create the conda environment from the provided YAML:

```bash
conda env create -f brainmaps_environment.yml
conda activate brainmaps-env
```

# Usage

You can use this in a python notebook. See the `Brain_Mapping_Pipeline.ipynb` for usage. 

## Importing Data

To use the class, you'll need to import a set of libraries. Example:

```python
import numpy as np
import matplotlib.pyplot as plt
import brainmaps as bmps
```

`brainmaps as bmps` is the library. This will need to be in the same directory that you're working in. numpy and matplotlib are useful for manipuating and viewing data, but are not directly needed for just the brainmaps side.

## Preprocessing

Typical preprocessing involves initializing an `Experiment` with imaging data, setting conditions, and verifying the setup. Each experiment comes with a zfirst flag - this refers to the order your image matrix is arranged in. brainmaps uses the z-axis first, and then the x,y, but many optputs from ImageJ/FIJI put the z last. Specifying that z-first is not true will reoder this so that it's correct.

brainmaps has a easy to use config method that will set up an experiment for you. To use, specify the Config method and give the method the filename, the condition, and the zfirst flag.

```python
# single file
bmps.Config(filepath="brainscans/SPF2_SF_LightDark_no5", condition="Surface_LD", zfirst=False)
```
 or 
 
 ```python
 # multiple files
 configs = [
    bmps.Config(filepath="brainscans/SPF2_SF_LightDark_no5", condition="Surface_LD", zfirst=False),
    bmps.Config(filepath="brainscans/SPF2_SF_LightDark_no5", condition="Pach_LD", zfirst=False),
]
 ```
 
 Each `filepath` has a _ 01 (and an optional _ 02) where _ 01 is the tERK and _ 02 is the pERK. So in this example, the path `SPF2_SF_LightDark_no5` would list two files in that directory called `SPF2_SF_LightDark_no5_01.nrrd` and `SPF2_SF_LightDark_no5_02.nrrd`
 
 
 
Once you have a config file established, you can set up an AnalysisSuite instance:


```python
configs = [
    bmps.Config(filepath="brainscans/SPF2_SF_LightDark_no5", condition="Surface_LD", zfirst=False),
    bmps.Config(filepath="brainscans/SPF2_SF_LightDark_no5", condition="Pach_LD", zfirst=False),
]

suite = bmps.AnalysisSuite.from_configs(configs)
```

This now creates an AnalsyisSuite instance with all of the brain images that you've loaded.

## Batch Config Setup

Oftentimes times you will have large datasets and making config files for these can be difficult and time-consuming. brainmaps comes equipped with a helper-function that allows you to build config files from larger datasets.

Assume you have an experiment that has young and aged brain, with both drug and placebo. Each condition has several .nrrd files. You may envision a directory structure like this:

```
root_folder/
├── young_control/
│   ├── brain_yc_one_01.nrrd
│   ├── brain_yc_one_02.nrrd
│   ├── brain_yc_two_01.nrrd
│   ├── brain_yc_two_02.nrrd
│
├── young_experimental/
│   ├── brain_ye_one_01.nrrd
│   ├── brain_ye_one_02.nrrd
│   ├── brain_ye_two_01.nrrd
│   ├── brain_ye_two_02.nrrd
│
├── aged_control/
│   ├── brain_ac_one_01.nrrd
│   ├── brain_ac_one_02.nrrd
│   ├── brain_ac_two_01.nrrd
│   ├── brain_ac_two_02.nrrd
│
├── aged_experimental/
│   ├── brain_ae_one_01.nrrd
│   ├── brain_ae_one_02.nrrd
│   ├── brain_ae_two_01.nrrd
│   ├── brain_ae_two_02.nrrd
```

You can make a config file using the following method: 

```
build_configs_from_directory("/data/root_folder", zfirst=False). 
```

The that will be a config file that look like this:

```
configs = [
    Config(
        filepath="/data/root_folder/young_control/brain_yc_01",
        condition="young_control",
        zfirst=False
    ),
    Config(
        filepath="/data/root_folder/young_control/brain_yc_02",
        condition="young_control",
        zfirst=False
    ),
    Config(
        filepath="/data/root_folder/young_control/brain_yc_03",
        condition="young_control",
        zfirst=False
    ),

    Config(
        filepath="/data/root_folder/young_experimental/brain_ye_01",
        condition="young_experimental",
        zfirst=False
    ),
    Config(
        filepath="/data/root_folder/young_experimental/brain_ye_02",
        condition="young_experimental",
        zfirst=False
    ),
    Config(
        filepath="/data/root_folder/young_experimental/brain_ye_03",
        condition="young_experimental",
        zfirst=False
    ),

    Config(
        filepath="/data/root_folder/aged_control/brain_ac_01",
        condition="aged_control",
        zfirst=False
    ),
    Config(
        filepath="/data/root_folder/aged_control/brain_ac_02",
        condition="aged_control",
        zfirst=False
    ),
    Config(
        filepath="/data/root_folder/aged_control/brain_ac_03",
        condition="aged_control",
        zfirst=False
    ),

    Config(
        filepath="/data/root_folder/aged_experimental/brain_ae_01",
        condition="aged_experimental",
        zfirst=False
    ),
    Config(
        filepath="/data/root_folder/aged_experimental/brain_ae_02",
        condition="aged_experimental",
        zfirst=False
    ),
    Config(
        filepath="/data/root_folder/aged_experimental/brain_ae_03",
        condition="aged_experimental",
        zfirst=False
    ),
]
You can also write a json with the flag: write_json="/path/configs.json"

build_configs_from_directory("/data/root_folder", zfirst=False, write_json="/path/configs.json)
```

To save time, we write a different constuctor called `build_configs_from_directory`. This will allow you to load in the root folder, and the package will load in each file with a condition named whatever the folder is called that that file is saved in:

```python
build_configs_from_directory("/data/root_folder", zfirst=False, write_json="/path/configs.json)
```

Note that you have the option to save a json config if you want.

## Preprocessing

#### Check
Before running the analysis, you'll need to ensure that the number of conditions, z-direction, and dimensions are all OK. There is a FLAG that will not let you proceed until this is done. It can be called implicitly later on, but its best to ensure all parameters check out here first:

```python
suite.check_anlysis_conditions()
```

## Permutation

The permutation is achieved by first setting up a permutation run and then executing it. You can first define the groups you want to compare and then make a permutation run using `prepare_for_permutation`:


```python
groups = ("Surface_LD", "Pach_LD")
# channel_mode options: "ch1", "ch2", "ndi_ch1", "ndi_ch2", "log_ratio_ch1", "log_ratio_ch2"
prepared = bmps.prepare_for_permutation(
    suite,
    groups=groups,
    channel_mode="ndi_ch2",      
    do_brightness_affine=True,     
    downsample = True,
    ds_factors = (2,2,2),
    ds_method = "local_mean"
)
```

There are several options here. First, you can set the channel mode. Your options here are:


* "ch1": Use channel 1 as-is
* "ch2": Use channel 2 as-is
* "ndi_ch1": Use normliazed differences for ch1: (ch1 - ch2) / (ch1 + ch2)  (emphasis: ch1 as the "signal")
* "ndi_ch2": Use normliazed differences for ch2: (ch2 - ch1) / (ch1 + ch2)  (emphasis: ch2 as the "signal")
* "log_ ratio_ch1": Use normliazed differences for ch1: log((ch1+eps) / (ch2+eps))
* "log_ ratio_ch2": Use normliazed differences for ch2: log((ch2+eps) / (ch1+eps))

Second, you can decide whether to normalize the brightness. This essentailly tries to make the average background level the same for all channels to control for large differences in laser power settings and optic when imaging. the flag `do_brightness_affine=True` will set this to yes, whereas `Fasle` will keep them as they are. 

Lastly, downsampling can help with the analysis load. `ds_factors` is a tuple with the downasmple factors for each axis - so `ds_factors = (2,2,2)` will cut each dimensuion in half.


Finally, `run_permutation` will run the specificed permutation:

```python 
res = bmps.run_permutation(prepared,groups = groups, n_perm = 2000, tail = "two-sided", alpha = 0.05)

```

You can specify the number of permutations, the tails, and the alpha.


## Accessing results

All results are saved in a `PermutationResults` class. You can print the results using print:

```python
print(res)
```

There are also a few different arrays in the PermutationResults:

```
effect_map: NDArray[Any]          # (Z, X, Y) observed difference in means: mean(g2) - mean(g1)
p_map: NDArray[Any]               # (Z, X, Y) permutation p-values
q_map: NDArray[Any]               # (Z, X, Y) BH-FDR q-values
sig_mask: NDArray[Any]            # (Z, X, Y) boolean, q <= alpha
```

For example, you can use matplotlib to view the effect map:

```python
# view 50th slice from effect map:
plt.imshow(res.effect_map[50,:,:])
plt.show()
```