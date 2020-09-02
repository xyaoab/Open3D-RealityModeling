# GPU Accelerated Robust Scene Reconstruction 
## -- The unofficial Open3D cuda branch

**We are pushing the development of the official cuda branch. Please be patient :)**

This is the **unofficial** cuda branch of [Open3D](http://www.open3d.org/), aiming at accelerating parallel operations like RGB-D Odometry and TSDF Integration.
Overall, this cuda pipeline can accelerate Open3D by a factor >10 for the scene reconstruction task. For a typical [lounge](http://qianyi.info/scenedata.html) scene,
the pipeline can finish reconstruction in 5 minutes (tested on a laptop with an Intel i7 CPU, 1070 GPU). As an offline system, it reaches around 5~10 fps on average.

For details, please refer to this [paper](http://dongwei.info/publications/open3d-gpu.pdf) accepted to IROS 2019.

## Build
- Apart from the [official depedencies](http://www.open3d.org/docs/compilation.html), the only additional requirement is [CUDA](https://developer.nvidia.com/cuda-downloads).
I haven't tested many distributions, but >= 8.0 should work. It has been tested on Ubuntu 16.04 and 18.04.
- The compilation would be the same as the official branch: `mkdir build && cd build && cmake .. && make -j4`
- During compilation, Eigen may complain about `half` precision floats in CUDA headers: specifically Eigen <= 3.3.5 against CUDA >= 9.0.
If you want to stick to the Eigen in `3rdparty`, a dirty workaround will be commenting out these lines in `3rdparty/Eigen/Eigen/Core`:
```
#include "src/Core/arch/CUDA/Half.h"
#include "src/Core/arch/CUDA/PacketMathHalf.h"
#include "src/Core/arch/CUDA/TypeCasting.h"
```

## Caveats
I would like put warning signs before usages:
- Currently there is NO python binding to the cuda implementations. We may fix this in the official cuda branch.
- Many directory strings are hard coded in the source code (I am lazy) -- you many have to change some lines of code to adapt to your own dataset directories.
This will definitely be fixed in the official cuda branch (maybe I will randomly fix some of them in this branch :-) ).
- Data management is very naive -- we may encounter memory overflow in some tasks (especially TSDF Integration, Marching Cubes, 3D feature extraction and brute force matching).
This is because I pre-allocate memory buffers with some hard coded buffer sizes. Again, you may want to manually change them in the sources.
If you are playing with devices with small GPU memory (like [Jetson](https://developer.nvidia.com/embedded/buy/jetson-tx2)),
you may have to significantly restrict the scene size or reduce resolution to make it work.
- There may be some discrepancies comparing to the official implementation, but most of the results should be identical.

## Usage
- There are some demo codes for separate functions in `examples/Cuda/Demo`. Most of the files are self-explanatory.
- The fully functional reconstruction system, almost identical to the [official implementation](http://www.open3d.org/docs/tutorial/ReconstructionSystem/index.html),
is available in `examples/Cuda/ReconstructionSystem`. To run `RunSystem`, please follow the [official document](http://www.open3d.org/docs/tutorial/ReconstructionSystem/capture_your_own_dataset.html#make-a-new-configuration-file), prepare datasets, and specify config files.

## Unofficial TODO
- Currently, the local loop-closures component is entirely disabled in order to remove the heavy opencv dependency.
Since this is only used for ORB feature extraction, I may implement (or import) it as an independent component.

## Contact
- To report problems, please use the [Discord channel](https://discordapp.com/invite/D35BGvn).
I will try to give some temporary workarounds and mark that as TODO in the official branch.

## License and Citation
This branch follows the [license of the official Open3D](https://github.com/intel-isl/Open3D/blob/master/LICENSE).

If you find this project useful, please cite our paper (in addition to the original Open3D citation):
```
@inproceedings{Dong-et-al-IROS-2019,
  title =        {{GPU} Accelerated Robust Scene Reconstruction},
  year =         {2019},
  author =       {Wei Dong and Jaesik Park and Yi Yang and Michael Kaess},
  booktitle =    {Proceedings of IEEE/RSJ International Conference on Intelligent Robots and Systems}
}
```
