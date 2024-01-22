import os
import app
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
import time

start = time.time()

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

def main():
    import_custom_nodes()
    with torch.inference_mode():
        print("\n\n\n Start")
        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_4 = checkpointloadersimple.load_checkpoint(
            ckpt_name="divineanimemix_V2.safetensors"
        )
        print("Did it load the checkpoint again?\n\n\n")
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

        ipadaptermodelloader = NODE_CLASS_MAPPINGS["IPAdapterModelLoader"]()
        ipadaptermodelloader_10 = get_value_at_index(ipadaptermodelloader.load_ipadapter_model(
            ipadapter_file="ip-adapter-full-face_sd15.bin"), 0)

        clipvisionloader = CLIPVisionLoader()
        clipvisionloader_12 = get_value_at_index(clipvisionloader.load_clip(
            clip_name="SD1.5\model.safetensors"), 0)

        controlnetloader = ControlNetLoader()
        controlnetloader_17 = get_value_at_index(controlnetloader.load_controlnet(
            control_net_name="control_v11p_sd15_lineart.pth"), 0)
        print("controlnetloader_26")
        controlnetloader_26 = get_value_at_index(controlnetloader.load_controlnet(
            control_net_name="control_v11p_sd15_scribble.pth"), 0)
        print("controlnetloader_26 end")
        clipseg_model_loader = NODE_CLASS_MAPPINGS["CLIPSeg Model Loader"]()
        clipseg_model_loader_35 = get_value_at_index(clipseg_model_loader.clipseg_model(
            model="CIDAS/clipseg-rd64-refined"), 0)

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

        controlnetloader_65 = get_value_at_index(controlnetloader.load_controlnet(
            control_net_name="control_v11p_sd15_lineart.pth"), 0)

        controlnetloader_78 = get_value_at_index(controlnetloader.load_controlnet(
            control_net_name="control_v11f1p_sd15_depth.pth"), 0)

        controlnetloader_109 = get_value_at_index(controlnetloader.load_controlnet(
            control_net_name="control_v2p_sd15_mediapipe_face.safetensors"), 0)


        clipseg_model_loader_129 = get_value_at_index(clipseg_model_loader.clipseg_model(
            model="CIDAS/clipseg-rd64-refined"), 0)


        cliptextencode = CLIPTextEncode()
        cliptextencode_6 = get_value_at_index(cliptextencode.encode(
            text="1girl, with name of Jennie Kim, solo, Korean, , closed mouth, aroused, black hair, black eyes, see through, sweating, dress, stockings, legs, earrings, gradient background, grey background, jewelry, long hair, huge breasts, hanging breasts, necklace, <lora:koreanDollLikeness:0.6>",
            clip=get_value_at_index(loraloader_146, 1),), 0)

        cliptextencode_7 = get_value_at_index(cliptextencode.encode(
            text="text, watermark", clip=get_value_at_index(loraloader_23, 1)), 0)

        loadimage = LoadImage()
        loadimage_13 = loadimage.load_image(image=r"C:\Users\zadka\OneDrive\Documents\ComfyUI_windows_portable\ComfyUI\input\ipadapter.png")

        loadimage_14 = loadimage.load_image(image=r"C:\Users\zadka\OneDrive\Documents\ComfyUI_windows_portable\ComfyUI\input\input.png")

        vaeencode = VAEEncode()
        vaeencode_15 = vaeencode.encode(         ## No Use?
            pixels=get_value_at_index(loadimage_14, 0),
            vae=get_value_at_index(checkpointloadersimple_4, 2),
        )
        text = NODE_CLASS_MAPPINGS["Text"]()
        text_42 = get_value_at_index(text.get_value(Text="Face"), 0)



        cliptextencode_62 = get_value_at_index(cliptextencode.encode(
            text="face shot, close shot, 1girl, with name of Jennie Kim, solo, Korean,closed mouth, aroused, black hair, black eyes, earrings, gradient background, grey background, jewelry, long hair, necklace, <lora:koreanDollLikeness:0.6>",
            clip=get_value_at_index(loraloader_144, 1),), 0)

        cliptextencode_63 = get_value_at_index(cliptextencode.encode(
            text="text, watermark", clip=get_value_at_index(loraloader_144, 1)), 0)

        lineartpreprocessor = NODE_CLASS_MAPPINGS["LineArtPreprocessor"]()
        lineartpreprocessor_91 = get_value_at_index(lineartpreprocessor.execute(
            resolution=512, image=get_value_at_index(loadimage_14, 0), coarse="enable"),0)

        controlnetapply = ControlNetApply()
        controlnetapply_16 = get_value_at_index(controlnetapply.apply_controlnet(
            strength=1,
            conditioning=cliptextencode_6,
            control_net=controlnetloader_17,
            image=lineartpreprocessor_91,
        ), 0)

        midas_depthmappreprocessor = NODE_CLASS_MAPPINGS["MiDaS-DepthMapPreprocessor"]()
        midas_depthmappreprocessor_71 = get_value_at_index(midas_depthmappreprocessor.execute(
            a=6.283185307179586,
            bg_threshold=0.1,
            resolution=512,
            image=get_value_at_index(loadimage_14, 0),
        ), 0)

        controlnetapply_25 = get_value_at_index(controlnetapply.apply_controlnet(
            strength=1,
            conditioning=controlnetapply_16,
            control_net=controlnetloader_26,
            image=midas_depthmappreprocessor_71,
        ), 0)

        getimagesize = NODE_CLASS_MAPPINGS["GetImageSize"]()
        getimagesize_153 = getimagesize.get_size(
            image=get_value_at_index(loadimage_14, 0)
        )

        emptylatentimage = EmptyLatentImage()
        emptylatentimage_155 = get_value_at_index(emptylatentimage.generate(
            width=get_value_at_index(getimagesize_153, 0),
            height=get_value_at_index(getimagesize_153, 1),
            batch_size=1,
        ), 0)

        ksampler = KSampler()
        ksampler_3 = get_value_at_index(ksampler.sample(
            seed=random.randint(1, 2**64),
            steps=6,
            cfg=1.5,
            sampler_name="lcm",
            scheduler="karras",
            denoise=1,
            model=get_value_at_index(loraloader_146, 0),
            positive=controlnetapply_25,
            negative=cliptextencode_7,
            latent_image=emptylatentimage_155,
        ), 0)

        vaedecode = VAEDecode()
        vaedecode_8 = get_value_at_index(vaedecode.decode(
            samples=ksampler_3,
            vae=get_value_at_index(checkpointloadersimple_4, 2),
        ), 0)

        clipseg_masking = NODE_CLASS_MAPPINGS["CLIPSeg Masking"]()
        clipseg_masking_41 = get_value_at_index(clipseg_masking.CLIPSeg_image(
            text=text_42,
            image=vaedecode_8,
            clipseg_model=clipseg_model_loader_35,
        ), 1)

        imagetocontrastmask = NODE_CLASS_MAPPINGS["ImageToContrastMask"]()
        imagetocontrastmask_37 = get_value_at_index(imagetocontrastmask.image_to_contrast_mask(
            low_threshold=250,
            high_threshold=150,
            blur_radius=2,
            image=clipseg_masking_41,
        ), 1)

        invertmask = NODE_CLASS_MAPPINGS["InvertMask"]()
        invertmask_38 = invertmask.invert(
            mask=imagetocontrastmask_37
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
            image=vaedecode_8,
        )

        vaeencode_67 = vaeencode.encode(
            pixels=get_value_at_index(image_crop_location_40, 0),
            vae=get_value_at_index(checkpointloadersimple_89, 2),
        )


        text_135 = text.get_value(Text="Face")

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

        for q in range(1):
            ipadapterapply_11 = ipadapterapply.apply_ipadapter(
                weight=0.75,
                noise=0.01,
                weight_type="original",
                start_at=0,
                end_at=0.85,
                unfold_batch=False,
                ipadapter=ipadaptermodelloader_10,
                clip_vision=clipvisionloader_12,
                image=get_value_at_index(loadimage_13, 0),
                model=get_value_at_index(loraloader_144, 0),
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


            mediapipe_facemeshpreprocessor_110 = mediapipe_facemeshpreprocessor.detect(
                max_faces=1,
                min_confidence=0.5,
                resolution=512,
                image=get_value_at_index(image_crop_location_40, 0),
            )

            controlnetapply_108 = controlnetapply.apply_controlnet(
                strength=1,
                conditioning=cliptextencode_62,
                control_net=controlnetloader_109,
                image=get_value_at_index(mediapipe_facemeshpreprocessor_110, 0),
            )

            lineartpreprocessor_127 = lineartpreprocessor.execute(
                resolution=512, image=get_value_at_index(image_crop_location_40, 0), coarse="enable"
            )

            controlnetapply_61 = controlnetapply.apply_controlnet(
                strength=1,
                conditioning=get_value_at_index(controlnetapply_108, 0),
                control_net=controlnetloader_65,
                image=get_value_at_index(lineartpreprocessor_127, 0),
            )

            zoe_depthmappreprocessor_149 = zoe_depthmappreprocessor.execute(
                resolution=960, image=get_value_at_index(image_crop_location_40, 0)
            )

            controlnetapply_76 = controlnetapply.apply_controlnet(
                strength=1,
                conditioning=get_value_at_index(controlnetapply_61, 0),
                control_net=controlnetloader_78,
                image=get_value_at_index(zoe_depthmappreprocessor_149, 0),
            )

            ksampler_58 = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=5,
                cfg=1.5,
                sampler_name="lcm",
                scheduler="karras",
                denoise=0.55,
                model=get_value_at_index(ipadapterapply_11, 0),
                positive=get_value_at_index(controlnetapply_76, 0),
                negative=cliptextencode_63,
                latent_image=get_value_at_index(vaeencode_67, 0),
            )

            vaedecode_64 = get_value_at_index(vaedecode.decode(
                samples=get_value_at_index(ksampler_58, 0),
                vae=get_value_at_index(checkpointloadersimple_89, 2),
            ), 0)


            imagecompositemasked_114 = imagecompositemasked.composite(
                x=get_value_at_index(mask_crop_region_39, 3),
                y=get_value_at_index(mask_crop_region_39, 2),
                resize_source=False,
                destination=vaedecode_8,
                source=vaedecode_64,
                mask=get_value_at_index(image_to_mask_51, 0),
            )


            clipseg_masking_134 = clipseg_masking.CLIPSeg_image(
                text=get_value_at_index(text_135, 0),
                image=get_value_at_index(loadimage_14, 0),
                clipseg_model=clipseg_model_loader_129,
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
            print(start - time.time())



#if __name__ == "__main__":


    # Add arguments for the image paths


    # Parse the arguments
    #args = parser.parse_args()
    #args.path1 = os.path.normpath(args.path1)
    #args.path2 = os.path.normpath(args.path2)

    #argsDict = {
    #    "ipadapter_input": args.path1,
    #    "image_input": args.path2
    #}
    #print(argsDict)
