# rgboverlay

## Synopsis
Create NIfTI RGB24 images by combining a base image with one or more overlay images

## Usage

```bash
rgboverlay rgboverlay [-h] [--version] -b FILE -br LO HI -ov FILE -ovr LO HI -ovc C -ova OVA [-out FILE]
```

## Options
- `-h`: show this help message and exit
- `--version`: show program's version number and exit

## Base image arguments:
- `-b`: base image filename
- `-br`: base image range from LO to HI inclusive. In the final rendered RGB 
image voxels with values below LO in the base image are clipped to LO i.e. 
they appear black and voxels with values above HI are clipped to HI i.e. 
they appear white

## Overlay image arguments (repeat the four arguments for each overlay):
- `-ov`: overlay image filename
- `-ovr`: overlay range from LO to HI inclusive. In the rendered RGB image(s) 
voxels with values below LO in the overlay image appear transparent i.e. just 
the base image is shown whereas voxels with values above HI are clipped to HI 
so they appear with the brightest colour in the chosen colour map
-  `-ovc`: colour map, choose from {red-yellow, blue-lightblue, red, blue, 
green, yellow, pink, cool}
- `-ova`: overlay opacity (percentage: 0-100)

## Output file arguments:
- `-out`: output filename

## Description
This package can be used to combine two or more NIfTI images into a blended
NIfTI RGB24 image, using the following steps:

1. clip the voxel values in the base image to the range supplied by `-br`
2. linearly scale the voxel values of the clipped base image to integers in 
the range 0-255
3. convert to the clipped and scaled base image to RGB with a greyscale colour 
map 
4. create a mask representing the voxels in the overlay image with values 
greater than `-ovr LO`
5. clip the voxel values in the first overlay image to the range supplied by 
`-ovr`
6. linearly scale the voxel values of the clipped overlay image to integers in 
the range 0-255
7. convert to the clipped and scaled overlay image to RGB using the colour 
map chosen with `-ovc`
8. blend the RGB base and overlay images using the opacity chosen with `-ova`  
using the following method:
    ```python
    base[mask] = base[mask] * (1 - alpha) + overlay[mask] * alpha
    ```
> [!NOTE]
> `alpha` is the fractional opacity whereas ova is supplied as a percentage

8. repeat steps 4 to 8 for each overlay supplied

## Installing
1. Create a new virtual environment in which to install `rgboverlay`:

    ```bash
    uv venv rgboverlay-venv
    ```
   
2. Activate the virtual environment:

    ```bash
    source rgboverlay-venv/bin/activate
    ```

4. Install using `uv pip`:
    ```bash
    uv pip install git+https://github.com/SWastling/rgboverlay.git
    ```
   
> [!TIP]
> You can also run `rgboverlay` without installing it using 
> [uvx](https://docs.astral.sh/uv/guides/tools/) i.e. with the 
>command `uvx --from  git+https://github.com/SWastling/rgboverlay.git rgboverlay`

## License
See [MIT license](./LICENSE)

## Author
[Stephen Wastling](mailto:stephen.wastling@nhs.net) 
