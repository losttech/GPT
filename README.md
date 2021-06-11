C# implementation of [GPT-2](https://en.wikipedia.org/wiki/GPT-2).

[![GPT on NuGet](https://img.shields.io/nuget/v/LostTech.TensorFlow.GPT)](https://www.nuget.org/packages/LostTech.TensorFlow.GPT/)

## Known issues

### CUDA out of host memory

There seems to be an issue with TensorFlow's default GPU memory allocator, that consumes more than needed.
In case you know you have enough RAM/GPU RAM, setting `TF_GPU_ALLOCATOR` environment variable to `cuda_malloc` might help.
