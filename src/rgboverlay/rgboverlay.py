import argparse
import importlib.metadata
import nibabel as nib
import numpy as np
import pathlib
import sys


SCRIPT_NAME = "rgboverlay"
__version__ = importlib.metadata.version(SCRIPT_NAME)


def check_overlay_args(args):
    """
    Check that user has specified all the necessary options for each overlay

    :param args: argparse object created by parse_args()
    :type args: argparse.Namespace
    """
    if len(args.ovr) != len(args.ov):
        raise RuntimeError(
            "Each overlay must be accompanied by an intensity range using the -ovr argument"
        )
    if len(args.ovc) != len(args.ov):
        raise RuntimeError(
            "Each overlay must be accompanied by a colour map using the -ovc argument"
        )
    if len(args.ova) != len(args.ov):
        raise RuntimeError(
            "Each overlay must be accompanied by an opacity using the -ova argument"
        )


def check_nifti_datatype(nii):
    """
    Check that NIfTI datatype is not complex or RGB etc...

    :param nii: NIfTI object
    :type nii: nib.nifti1.Nifti1Image
    """

    if int(nii.header["datatype"]) not in [
        1,
        2,
        4,
        8,
        16,
        64,
        256,
        512,
        768,
        1024,
        1280,
        1536,
    ]:
        #    1 = binary (1 bit/voxel)
        #    2 = unsigned char (8 bits/voxel)
        #    4 = signed short (16 bits/voxel)
        #    8 = signed int (32 bits/voxel)
        #   16 = float (32 bits/voxel)
        #   64 = double (64 bits/voxel)
        #  256 = signed char (8 bits/voxel)
        #  512 = unsigned short (16 bits/voxel)
        #  768 = unsigned int (32 bits/voxel)
        # 1024 = long long (64 bits/voxel)
        # 1280 = unsigned long long (64 bits/voxel)
        # 1536 = long double (128 bits/voxel)
        raise TypeError(
            "NIfTI datatype must be 1, 2, 4, 8, 16, 64, 256, 512, 768, 1024, 1280 or 1536"
        )


def check_shape_and_orientation(a_obj, b_obj):
    """
    Compare the affine and matrix size in the header of two NIfTI files

    :param a_obj: first NIfTI object
    :type a_obj: nib.nifti1.Nifti1Image
    :param b_obj: second NIfTI object
    :type b_obj: nib.nifti1.Nifti1Image
    :return: True is matching, False if not
    :rtype: bool
    """

    a_affine = a_obj.header.get_best_affine()
    a_shape = a_obj.header.get_data_shape()

    b_affine = b_obj.header.get_best_affine()
    b_shape = b_obj.header.get_data_shape()

    if np.allclose(a_affine, b_affine, atol=1e-4) and (a_shape == b_shape):
        return True
    else:
        return False


def create_rgb_mask(im, lo):
    """
    Create a mask with three channels (R, G, B), based on all the voxel values
    in the image greater than lo

    :param im: Input image
    :type im: np.ndarray
    :param lo: lower limit for voxel value clipping
    :type lo: float
    :return: mask
    :rtype: np.ndarray
    """
    im = np.nan_to_num(im)
    mask = im > lo
    mask = np.stack((mask, mask, mask), axis=-1)

    return mask


def colourmap_lut(cmap_name):
    """
    Generate colourmap look-up-table that match those supplied with FSleyes

    :param cmap_name: name of colourmap
    :type cmap_name: str
    :return: colourmap
    :rtype: np.ndarray
    """

    if cmap_name == "gray":
        # 255 element array with values from 0 to 255
        red = np.arange(0, 256)
        green = np.arange(0, 256)
        blue = np.arange(0, 256)
    elif cmap_name == "red-yellow":
        red = np.full(256, 255)
        green = np.arange(0, 256)
        blue = np.zeros(256)
    elif cmap_name == "blue-lightblue":
        red = np.zeros(256)
        green = np.arange(0, 256)
        blue = np.full(256, 255)
    elif cmap_name == "red":
        red = np.linspace(100, 255, 256)
        green = np.zeros(256)
        blue = np.zeros(256)
    elif cmap_name == "blue":
        red = np.zeros(256)
        green = np.zeros(256)
        blue = np.linspace(100, 255, 256)
    elif cmap_name == "green":
        red = np.zeros(256)
        green = np.linspace(100, 255, 256)
        blue = np.zeros(256)
    elif cmap_name == "yellow":
        red = np.linspace(100, 255, 256)
        green = np.linspace(100, 255, 256)
        blue = np.zeros(256)
    elif cmap_name == "pink":
        red = np.full(256, 255)
        green = np.linspace(100, 255, 256)
        blue = np.linspace(100, 255, 256)
    elif cmap_name == "cool":
        red = np.arange(0, 256)
        green = np.arange(256, 0, -1)
        blue = np.full(256, 255)
    else:
        raise ValueError("colour map name not recognised")

    return np.column_stack((red, green, blue)).round().astype(np.uint8)


def convert_to_4d_rgb(im, lo, hi, cmap_lut):
    """
    Convert a 3D array to a 4D array (x,y,z,RGB) using a colour map

    :param im: Input image
    :type im: np.ndarray
    :param lo: lower limit for voxel value clipping
    :type lo: float
    :param hi: upper limit for voxel value clipping
    :type hi: float
    :param cmap_lut: colour map look-up table (255,3) array
    :type cmap_lut: np.ndarray
    :return: 4D array (x,y,z,RGB)
    :rtype: np.ndarray[np.uint8]
    """

    # replace NaN values with 0 in input array
    im = np.nan_to_num(im)

    # clip (limit) the values in the input array i.e. values less than lo
    # become lo and values greater than hi become hi.
    im = np.clip(im, lo, hi)

    if np.ptp(im) == 0.0:
        raise RuntimeError(
            "The intensity range in the image after clipping with LO HI values is zero"
        )

    # scale, round to integer and then cast to unsigned 8-bit int
    im = np.rint(255 * (im - np.min(im)) / np.ptp(im)).astype(np.uint8)

    # apply colour look up table
    im_4d_rgb = np.take(cmap_lut, im, axis=0)

    return im_4d_rgb


def blend(base_image, overlay, mask, alpha):
    """
    Add overlay, with transparency alpha, to base image

    :param base_image: base image
    :type base_image: np.ndarray
    :param overlay: overlay image
    :type overlay: np.ndarray
    :param mask: mask image true in regions where overlay > lower threshold
    :type mask: np.ndarray[bool]
    :param alpha: overlay transparency [0:1]
    :type alpha: float
    :return: blended base and overlay image
    :rtype: np.ndarray
    """

    base_image[mask] = base_image[mask] * (1 - alpha) + overlay[mask] * alpha

    return base_image


def cast2rgb(data_array):
    """
    Convert a 4D array (x,y,z,RGB) to a custom NIfTI RGB datatype as per
    Matthew Brett's instructions at
    https://mail.python.org/pipermail/neuroimaging/2016-November/001231.html

    :param data_array: 4D array (x,y,z,RGB)
    :type data_array: np.ndarray
    :return: data_array: data array with RGB type
    :rtype: [("R", np.uint8), ("G", np.uint8), ("B", np.uint8)]
    """
    rgb_dtype = np.dtype([("R", np.uint8), ("G", np.uint8), ("B", np.uint8)])

    output_shape = data_array.shape[0:3]

    if data_array.ndim != 4:
        raise ValueError("input data array should have 4 dimensions")

    if data_array.shape[3] != 3:
        raise ValueError("input data array 4th dimension must have size 3 (RGB)")

    return data_array.copy().view(dtype=rgb_dtype).reshape(output_shape)


def main():
    parser = argparse.ArgumentParser(
        description="Create NIfTI RGB24 images by "
        "combining a base image "
        "with one or more overlay "
        "images"
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    base_group = parser.add_argument_group("base image arguments")
    base_group.add_argument(
        "-b",
        type=pathlib.Path,
        metavar="FILE",
        help="base image filename",
        required=True,
    )
    base_group.add_argument(
        "-br",
        type=float,
        nargs=2,
        metavar=("LO", "HI"),
        help="base image range from LO to HI inclusive. "
        "In the final rendered RGB image voxels with "
        "values below LO in the base image are "
        "clipped to LO i.e. they appear black and "
        "voxels with values above HI are clipped to "
        "HI i.e. they appear white",
        required=True,
    )

    overlay_group = parser.add_argument_group(
        "overlay image arguments (repeat the four arguments for each overlay)"
    )
    overlay_group.add_argument(
        "-ov",
        type=pathlib.Path,
        metavar="FILE",
        action="append",
        help="overlay image filename",
        required=True,
    )
    overlay_group.add_argument(
        "-ovr",
        type=float,
        nargs=2,
        metavar=("LO", "HI"),
        action="append",
        help="overlay range from LO to HI inclusive. In "
        "the rendered RGB image(s) voxels with "
        "values below LO in the overlay image "
        "appear transparent i.e. just the base "
        "image is shown whereas voxels with values "
        "above HI are clipped to HI so they appear "
        "with the brightest colour in the chosen "
        "colour map",
        required=True,
    )
    overlay_group.add_argument(
        "-ovc",
        metavar="C",
        type=str,
        action="append",
        help="colour map, choose from {%(choices)s}",
        choices=[
            "red-yellow",
            "blue-lightblue",
            "red",
            "blue",
            "green",
            "yellow",
            "pink",
            "cool",
        ],
        required=True,
    )
    overlay_group.add_argument(
        "-ova",
        type=int,
        action="append",
        help="overlay opacity (percentage: 0-100)",
        required=True,
    )

    combined_overlay_group = parser.add_argument_group("output file arguments")
    combined_overlay_group.add_argument(
        "-out", metavar="FILE", type=pathlib.Path, help="output filename"
    )

    if len(sys.argv) == 1:
        sys.argv.append("-h")

    args = parser.parse_args()

    try:
        check_overlay_args(args)
    except RuntimeError as err:
        sys.stderr.write("Error: %s\n" % err)
        sys.exit(1)

    base_fp = args.b.resolve()
    try:
        base_nii = nib.load(base_fp)
    except (FileNotFoundError, nib.filebasedimages.ImageFileError):
        sys.stderr.write("Error: unable to load %s\n" % base_fp)
        sys.exit(1)

    try:
        check_nifti_datatype(base_nii)
    except TypeError as te:
        sys.stderr.write("Error: %s\n" % te)
        sys.exit(1)

    base = base_nii.get_fdata()
    affine = base_nii.header.get_best_affine()
    try:
        base_4d_rgb = convert_to_4d_rgb(
            base, args.br[0], args.br[1], colourmap_lut("gray")
        )
    except RuntimeError as err:
        sys.stderr.write(
            "Error: %s, check -br values are appropriate given "
            "the range of voxel values in the base image\n" % err
        )
        sys.exit(1)

    for count, ov_fp in enumerate(args.ov):
        overlay_fp = ov_fp.resolve()
        ovr = args.ovr[count]
        ovc = args.ovc[count]
        ova = args.ova[count] / 100

        try:
            overlay_nii = nib.load(overlay_fp)
        except (FileNotFoundError, nib.filebasedimages.ImageFileError):
            sys.stderr.write("Error: unable to load %s\n" % overlay_fp)
            sys.exit(1)

        try:
            check_nifti_datatype(overlay_nii)
        except TypeError as te:
            sys.stderr.write("Error: %s\n" % te)
            sys.exit(1)

        overlay = overlay_nii.get_fdata()

        if not check_shape_and_orientation(base_nii, overlay_nii):
            sys.stderr.write(
                "Error: Base and overlay NIfTIs have mismatched geometry\n"
            )
            sys.exit(1)

        try:
            overlay_4d_rgb = convert_to_4d_rgb(
                overlay, ovr[0], ovr[1], colourmap_lut(ovc)
            )
        except RuntimeError as err:
            sys.stderr.write(
                "Error: %s, check -ovr values are appropriate "
                "given the range of voxel values in the overlay "
                "image\n" % err
            )
            sys.exit(1)

        mask = create_rgb_mask(overlay, ovr[0])

        if count == 0:
            output_4d_rgb = blend(base_4d_rgb, overlay_4d_rgb, mask, ova)
        else:
            output_4d_rgb = blend(output_4d_rgb, overlay_4d_rgb, mask, ova)

    nii_obj = nib.nifti1.Nifti1Image(cast2rgb(output_4d_rgb), affine)
    print("adding sform and qform")
    nii_obj.set_qform(affine, code=2)
    nii_obj.set_sform(affine, code=2)
    nii_obj.to_filename(args.out.resolve())


if __name__ == "__main__":  # pragma: no cover
    main()
