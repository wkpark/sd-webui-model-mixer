## Checkpoint Model Mixer extension
This is a simple but useful model merger/mixer.

It was created when I was looking for a way to use [SuperMerger](https://github.com/hako-mikan/sd-webui-supermerger) with other extensions in the txt2img tab. It doesn't have all the features of SuperMerger, but it inherits the core functionality.

## pros
- Supports merging multiple models sequentially (up to 5).
- Specialises in block-level merging to make block merging easier.
- Can be used without saving the merged model like [SuperMerger](https://github.com/hako-mikan/sd-webui-supermerger).
- Supports saving already merged models.
- Merge information is saved in the image, and when the image is loaded, it will retrieve the settings used in the merged model.
- Support XYZ plots

## cons
- Doesn't support many of the calculation methods supported by [SuperMerger](https://github.com/hako-mikan/sd-webui-supermerger).

## screenshot
![image](https://github.com/wkpark/sd-webui-model-mixer/assets/232347/472d7e22-91ba-4690-8fb2-dbe04245f22a)
