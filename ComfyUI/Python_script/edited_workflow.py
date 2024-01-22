import os
import random
import sys
import time
import uuid
from io import BytesIO
from typing import Sequence, Mapping, Any, Union, Generator, Tuple, List
import torch
import argparse
import asyncio
from azure.storage.blob import BlobServiceClient
import numpy as np
from PIL import Image


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    # sys.path.append("D:\ComfyUI\ComfyUI")
    # print(sys.path)
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    from main import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_custom_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_custom_nodes()


from nodes import (
    SaveImage,
    VAEEncode,
    KSampler,
    NODE_CLASS_MAPPINGS,
    CLIPVisionLoader,
    CheckpointLoaderSimple,
    VAEDecode,
    ControlNetLoader,
    CLIPTextEncode,
    LoadImage,
    LoraLoader,
    ControlNetApply,
    EmptyLatentImage,
)


async def main(data_dict, queue: asyncio.Queue) -> None:
    import_custom_nodes()
    with torch.inference_mode():
        time_0 = time.time()
        yield "Importing Models..."
        await asyncio.sleep(1)
        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_4 = checkpointloadersimple.load_checkpoint(
            ckpt_name="divineanimemix_V2.safetensors"
        )

        loraloader = LoraLoader()
        loraloader_23 = loraloader.load_lora(
            lora_name="koreanDollLikeness.safetensors",
            strength_model=0.63,
            strength_clip=0.6,
            model=get_value_at_index(checkpointloadersimple_4, 0),
            clip=get_value_at_index(checkpointloadersimple_4, 1),
        )

        loraloader_146 = loraloader.load_lora(
            lora_name="LCM_LoRA_Weights_SD15.safetensors",
            strength_model=1,
            strength_clip=1,
            model=get_value_at_index(loraloader_23, 0),
            clip=get_value_at_index(loraloader_23, 1),
        )

        cliptextencode = CLIPTextEncode()
        cliptextencode_6 = cliptextencode.encode(
            text=data_dict["prompt_1"],
            clip=get_value_at_index(loraloader_146, 1),
        )

        cliptextencode_7 = cliptextencode.encode(
            text="text, watermark", clip=get_value_at_index(loraloader_23, 1)
        )

        ipadaptermodelloader = NODE_CLASS_MAPPINGS["IPAdapterModelLoader"]()
        ipadaptermodelloader_10 = ipadaptermodelloader.load_ipadapter_model(
            ipadapter_file="ip-adapter-full-face_sd15.bin"
        )

        clipvisionloader = CLIPVisionLoader()
        clipvisionloader_12 = clipvisionloader.load_clip(
            clip_name="SD1.5\model.safetensors"
        )

        loadimage = LoadImage()
        # ip adapter
        loadimage_13 = loadimage.load_image(image=data_dict["ipadapter_input"])

        loadimage_14 = loadimage.load_image(image=data_dict["image_input"])

        vaeencode = VAEEncode()
        vaeencode_15 = vaeencode.encode(
            pixels=get_value_at_index(loadimage_14, 0),
            vae=get_value_at_index(checkpointloadersimple_4, 2),
        )

        controlnetloader = ControlNetLoader()
        controlnetloader_17 = controlnetloader.load_controlnet(
            control_net_name="control_v11p_sd15_lineart.pth"
        )

        controlnetloader_26 = controlnetloader.load_controlnet(
            control_net_name="control_v11p_sd15_scribble.pth"
        )

        clipseg_model_loader = NODE_CLASS_MAPPINGS["CLIPSeg Model Loader"]()
        clipseg_model_loader_35 = clipseg_model_loader.clipseg_model(
            model="CIDAS/clipseg-rd64-refined"
        )

        text = NODE_CLASS_MAPPINGS["Text"]()
        text_42 = text.get_value(Text="Face")

        checkpointloadersimple_89 = checkpointloadersimple.load_checkpoint(
            ckpt_name="divineelegancemix_V8.safetensors"
        )

        loraloader_90 = loraloader.load_lora(
            lora_name="koreanDollLikeness.safetensors",
            strength_model=0.62,
            strength_clip=1,
            model=get_value_at_index(checkpointloadersimple_89, 0),
            clip=get_value_at_index(checkpointloadersimple_89, 1),
        )

        loraloader_144 = loraloader.load_lora(
            lora_name="LCM_LoRA_Weights_SD15.safetensors",
            strength_model=1,
            strength_clip=1,
            model=get_value_at_index(loraloader_90, 0),
            clip=get_value_at_index(loraloader_90, 1),
        )

        cliptextencode_62 = cliptextencode.encode(
            text=data_dict["prompt_2"],
            clip=get_value_at_index(loraloader_144, 1),
        )

        cliptextencode_63 = cliptextencode.encode(
            text="text, watermark", clip=get_value_at_index(loraloader_144, 1)
        )

        controlnetloader_65 = controlnetloader.load_controlnet(
            control_net_name="control_v11p_sd15_lineart.pth"
        )

        lineartpreprocessor = NODE_CLASS_MAPPINGS["LineArtPreprocessor"]()
        lineartpreprocessor_91 = lineartpreprocessor.execute(
            resolution=512, image=get_value_at_index(loadimage_14, 0), coarse="disable"
        )

        controlnetapply = ControlNetApply()
        controlnetapply_16 = controlnetapply.apply_controlnet(
            strength=1,
            conditioning=get_value_at_index(cliptextencode_6, 0),
            control_net=get_value_at_index(controlnetloader_17, 0),
            image=get_value_at_index(lineartpreprocessor_91, 0),
        )

        midas_depthmappreprocessor = NODE_CLASS_MAPPINGS["MiDaS-DepthMapPreprocessor"]()
        midas_depthmappreprocessor_71 = midas_depthmappreprocessor.execute(
            a=6.283185307179586,
            bg_threshold=0.1,
            resolution=512,
            image=get_value_at_index(loadimage_14, 0),
        )

        controlnetapply_25 = controlnetapply.apply_controlnet(
            strength=1,
            conditioning=get_value_at_index(controlnetapply_16, 0),
            control_net=get_value_at_index(controlnetloader_26, 0),
            image=get_value_at_index(midas_depthmappreprocessor_71, 0),
        )
        yield "Running the models..."
        await asyncio.sleep(1)
        getimagesize = NODE_CLASS_MAPPINGS["GetImageSize"]()
        getimagesize_153 = getimagesize.get_size(
            image=get_value_at_index(loadimage_14, 0)
        )

        emptylatentimage = EmptyLatentImage()
        emptylatentimage_155 = emptylatentimage.generate(
            width=get_value_at_index(getimagesize_153, 0),
            height=get_value_at_index(getimagesize_153, 1),
            batch_size=1,
        )

        ksampler = KSampler()
        ksampler_3 = ksampler.sample(
            seed=random.randint(1, 2 ** 64),
            steps=6,
            cfg=1.5,
            sampler_name="lcm",
            scheduler="karras",
            denoise=1,
            model=get_value_at_index(loraloader_146, 0),
            positive=get_value_at_index(controlnetapply_25, 0),
            negative=get_value_at_index(cliptextencode_7, 0),
            latent_image=get_value_at_index(emptylatentimage_155, 0),
        )

        vaedecode = VAEDecode()
        vaedecode_8 = vaedecode.decode(
            samples=get_value_at_index(ksampler_3, 0),
            vae=get_value_at_index(checkpointloadersimple_4, 2),
        )

        clipseg_masking = NODE_CLASS_MAPPINGS["CLIPSeg Masking"]()
        clipseg_masking_41 = clipseg_masking.CLIPSeg_image(
            text=get_value_at_index(text_42, 0),
            image=get_value_at_index(vaedecode_8, 0),
            clipseg_model=get_value_at_index(clipseg_model_loader_35, 0),
        )

        imagetocontrastmask = NODE_CLASS_MAPPINGS["ImageToContrastMask"]()
        imagetocontrastmask_37 = imagetocontrastmask.image_to_contrast_mask(
            low_threshold=250,
            high_threshold=150,
            blur_radius=2,
            image=get_value_at_index(clipseg_masking_41, 1),
        )

        invertmask = NODE_CLASS_MAPPINGS["InvertMask"]()
        invertmask_38 = invertmask.invert(
            mask=get_value_at_index(imagetocontrastmask_37, 1)
        )

        mask_crop_region = NODE_CLASS_MAPPINGS["Mask Crop Region"]()
        mask_crop_region_39 = mask_crop_region.mask_crop_region(
            padding=24,
            region_type="dominant",
            mask=get_value_at_index(invertmask_38, 0),
        )

        image_crop_location = NODE_CLASS_MAPPINGS["Image Crop Location"]()
        image_crop_location_40 = image_crop_location.image_crop_location(
            top=get_value_at_index(mask_crop_region_39, 2),
            left=get_value_at_index(mask_crop_region_39, 3),
            right=get_value_at_index(mask_crop_region_39, 4),
            bottom=get_value_at_index(mask_crop_region_39, 5),
            image=get_value_at_index(vaedecode_8, 0),
        )

        vaeencode_67 = vaeencode.encode(
            pixels=get_value_at_index(image_crop_location_40, 0),
            vae=get_value_at_index(checkpointloadersimple_89, 2),
        )

        upscalemodelloader = NODE_CLASS_MAPPINGS["UpscaleModelLoader"]()
        upscalemodelloader_75 = upscalemodelloader.load_model(
            model_name="4x_fatal_Anime_500000_G.pth"
        )

        controlnetloader_78 = controlnetloader.load_controlnet(
            control_net_name="control_v11f1p_sd15_depth.pth"
        )

        controlnetloader_109 = controlnetloader.load_controlnet(
            control_net_name="control_v2p_sd15_mediapipe_face.safetensors"
        )

        upscalemodelloader_116 = upscalemodelloader.load_model(
            model_name="4x_fatal_Anime_500000_G.pth"
        )


        clipseg_model_loader_129 = clipseg_model_loader.clipseg_model(
            model="CIDAS/clipseg-rd64-refined"
        )

        text_135 = text.get_value(Text="Face")

        upscalemodelloader_138 = upscalemodelloader.load_model(
            model_name="4xUltrasharp_4xUltrasharpV10.pt"
        )
        yield "Upscalling Results..."
        await asyncio.sleep(1)
        ipadapterapply = NODE_CLASS_MAPPINGS["IPAdapterApply"]()
        masktoimage = NODE_CLASS_MAPPINGS["MaskToImage"]()
        growmask = NODE_CLASS_MAPPINGS["GrowMask"]()
        mask_morphology = NODE_CLASS_MAPPINGS["Mask Morphology"]()
        blur = NODE_CLASS_MAPPINGS["Blur"]()
        image_to_mask = NODE_CLASS_MAPPINGS["Image To Mask"]()
        mediapipe_facemeshpreprocessor = NODE_CLASS_MAPPINGS[
            "MediaPipe-FaceMeshPreprocessor"
        ]()
        zoe_depthmappreprocessor = NODE_CLASS_MAPPINGS["Zoe-DepthMapPreprocessor"]()
        cut_by_mask = NODE_CLASS_MAPPINGS["Cut By Mask"]()
        imagecompositemasked = NODE_CLASS_MAPPINGS["ImageCompositeMasked"]()
        saveimage = SaveImage()
        leres_depthmappreprocessor = NODE_CLASS_MAPPINGS["LeReS-DepthMapPreprocessor"]()


        generated_images = []
        for q in range(1):
            print("\n\n@@@@@@@@@@@this is round: ", q, "at time: \n", time.time() - time_0, "\n\n", )
            yield "Refining Results..."
            await asyncio.sleep(1)
            ipadapterapply_11 = ipadapterapply.apply_ipadapter(
                weight=0.75,
                noise=0.01,
                weight_type="original",
                start_at=0,
                end_at=0.85,
                # unfold_batch=False,
                ipadapter=get_value_at_index(ipadaptermodelloader_10, 0),
                clip_vision=get_value_at_index(clipvisionloader_12, 0),
                image=get_value_at_index(loadimage_13, 0),
                model=get_value_at_index(loraloader_144, 0),
            )

            masktoimage_44 = masktoimage.mask_to_image(
                mask=get_value_at_index(mask_crop_region_39, 0)
            )

            growmask_47 = growmask.expand_mask(
                expand=12,
                tapered_corners=True,
                mask=get_value_at_index(mask_crop_region_39, 0),
            )

            masktoimage_48 = masktoimage.mask_to_image(
                mask=get_value_at_index(growmask_47, 0)
            )

            mask_morphology_50 = mask_morphology.morph(
                distance=5, op="open", image=get_value_at_index(masktoimage_48, 0)
            )

            blur_49 = blur.blur(
                radius=15,
                sigma_factor=1.01,
                image=get_value_at_index(mask_morphology_50, 0),
            )

            image_to_mask_51 = image_to_mask.convert(
                method="intensity", image=get_value_at_index(blur_49, 0)
            )

            masktoimage_52 = masktoimage.mask_to_image(
                mask=get_value_at_index(image_to_mask_51, 0)
            )

            mediapipe_facemeshpreprocessor_110 = mediapipe_facemeshpreprocessor.detect(
                max_faces=1,
                min_confidence=0.5,
                resolution=512,
                image=get_value_at_index(image_crop_location_40, 0),
            )

            controlnetapply_108 = controlnetapply.apply_controlnet(
                strength=1,
                conditioning=get_value_at_index(cliptextencode_62, 0),
                control_net=get_value_at_index(controlnetloader_109, 0),
                image=get_value_at_index(mediapipe_facemeshpreprocessor_110, 0),
            )

            lineartpreprocessor_127 = lineartpreprocessor.execute(
                resolution=512, image=get_value_at_index(image_crop_location_40, 0), coarse="disable"
            )

            controlnetapply_61 = controlnetapply.apply_controlnet(
                strength=1,
                conditioning=get_value_at_index(controlnetapply_108, 0),
                control_net=get_value_at_index(controlnetloader_65, 0),
                image=get_value_at_index(lineartpreprocessor_127, 0),
            )

            zoe_depthmappreprocessor_149 = zoe_depthmappreprocessor.execute(
                resolution=960, image=get_value_at_index(image_crop_location_40, 0)
            )

            controlnetapply_76 = controlnetapply.apply_controlnet(
                strength=1,
                conditioning=get_value_at_index(controlnetapply_61, 0),
                control_net=get_value_at_index(controlnetloader_78, 0),
                image=get_value_at_index(zoe_depthmappreprocessor_149, 0),
            )

            ksampler_58 = ksampler.sample(
                seed=random.randint(1, 2 ** 64),
                steps=5,
                cfg=1.5,
                sampler_name="lcm",
                scheduler="karras",
                denoise=0.55,
                model=get_value_at_index(ipadapterapply_11, 0),
                positive=get_value_at_index(controlnetapply_76, 0),
                negative=get_value_at_index(cliptextencode_63, 0),
                latent_image=get_value_at_index(vaeencode_67, 0),
            )

            vaedecode_64 = vaedecode.decode(
                samples=get_value_at_index(ksampler_58, 0),
                vae=get_value_at_index(checkpointloadersimple_89, 2),
            )

            cut_by_mask_55 = cut_by_mask.cut(
                force_resize_width=0,
                force_resize_height=0,
                image=get_value_at_index(vaedecode_64, 0),
                mask=get_value_at_index(blur_49, 0),
            )

            midas_depthmappreprocessor_92 = midas_depthmappreprocessor.execute(
                a=6.8,
                bg_threshold=0.05,
                resolution=640,
                image=get_value_at_index(image_crop_location_40, 0),
            )

            imagecompositemasked_114 = imagecompositemasked.composite(
                x=get_value_at_index(mask_crop_region_39, 3),
                y=get_value_at_index(mask_crop_region_39, 2),
                resize_source=False,
                destination=get_value_at_index(vaedecode_8, 0),
                source=get_value_at_index(vaedecode_64, 0),
                mask=get_value_at_index(image_to_mask_51, 0),
            )


            saveimage_115 = saveimage.save_images(
                filename_prefix="ComfyUI",
                images=get_value_at_index(imagecompositemasked_114, 0),
            )
            generated_images.append(saveimage_115)

            clipseg_masking_134 = clipseg_masking.CLIPSeg_image(
                text=get_value_at_index(text_135, 0),
                image=get_value_at_index(loadimage_14, 0),
                clipseg_model=get_value_at_index(clipseg_model_loader_129, 0),
            )

            imagetocontrastmask_130 = imagetocontrastmask.image_to_contrast_mask(
                low_threshold=250,
                high_threshold=150,
                blur_radius=2,
                image=get_value_at_index(clipseg_masking_134, 1),
            )

            invertmask_131 = invertmask.invert(
                mask=get_value_at_index(imagetocontrastmask_130, 1)
            )

            mask_crop_region_132 = mask_crop_region.mask_crop_region(
                padding=24,
                region_type="dominant",
                mask=get_value_at_index(invertmask_131, 0),
            )

            image_crop_location_133 = image_crop_location.image_crop_location(
                top=get_value_at_index(mask_crop_region_132, 2),
                left=get_value_at_index(mask_crop_region_132, 3),
                right=get_value_at_index(mask_crop_region_132, 4),
                bottom=get_value_at_index(mask_crop_region_132, 5),
                image=get_value_at_index(loadimage_14, 0),
            )

            leres_depthmappreprocessor_147 = leres_depthmappreprocessor.execute(
                rm_nearest=0,
                rm_background=0,
                resolution=768,
                image=get_value_at_index(image_crop_location_40, 0),
                boost="enable" if "boost" in locals() else "disable"
            )
            yield "Images Generated."
            await asyncio.sleep(1)
        await queue.put(generated_images)
        await queue.put(None)  # Signal that the generator has completed
























# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Load an image from a specified path.')

#     # Add arguments for the image paths
#     parser.add_argument('-p1', '--path1', type=str,
#                         required=True, help='Path to the first image file')
#     parser.add_argument('-p2', '--path2', type=str,
#                         required=True, help='Path to the second image file')

#     # Parse the arguments
#     args = parser.parse_args()
#     args.path1 = os.path.normpath(args.path1)
#     args.path2 = os.path.normpath(args.path2)

#     argsDict = {
#         "ipadapter_input": args.path1,
#         "image_input":args.path2
#     }

#     # Call main function with the provided image paths
#     main(argsDict)
