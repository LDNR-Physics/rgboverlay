import argparse
import nibabel as nib
import numpy as np
import pathlib
import pytest
import importlib.metadata

import rgboverlay.rgboverlay as rgboverlay

THIS_DIR = pathlib.Path(__file__).resolve().parent
TEST_DATA_DIR = THIS_DIR / "test_data"

SCRIPT_NAME = "rgboverlay"
SCRIPT_USAGE = (
    f"usage: {SCRIPT_NAME} [-h] [--version] -b FILE -br LO HI -ov FILE -ovr LO HI"
)
__version__ = importlib.metadata.version(SCRIPT_NAME)


def perror(r_fp, t_fp):
    """
    Calculate the percentage error between two nifti files; a reference and
    a test

    Based on test used in FSL Evaluation and Example Data Suite (FEEDS)

    :param r_fp: reference file
    :type r_fp: pathlib.Path
    :param t_fp: test file
    :type t_fp: pathlib.Path
    return: percentage error of r and t
    type: float
    """

    r_obj = nib.load(r_fp)
    # nibabel defaults to float64 so we need to explicitly check for RGB type
    r_type = r_obj.get_data_dtype()
    if r_type == [("R", "u1"), ("G", "u1"), ("B", "u1")]:
        r = r_obj.dataobj.get_unscaled()
        shape_4d = r.shape + (3,)
        r = r.copy().view(dtype="u1").reshape(shape_4d)
    else:
        r = r_obj.get_fdata()

    t_obj = nib.load(t_fp)
    t_type = t_obj.get_data_dtype()
    if t_type == [("R", "u1"), ("G", "u1"), ("B", "u1")]:
        t = t_obj.dataobj.get_unscaled()
        shape_4d = t.shape + (3,)
        t = t.copy().view(dtype="u1").reshape(shape_4d)
    else:
        t = t_obj.get_fdata()

    return 100.0 * np.sqrt(np.mean(np.square(r - t)) / np.mean(np.square(r)))


@pytest.mark.parametrize(
    "ref_fn, test_fn, expected_output ",
    [
        ("OneHundred.nii.gz", "OneHundredOne.nii.gz", 1.0),
        ("OneHundred.nii.gz", "NinetyNine.nii.gz", 1.0),
        ("OneHundred.nii.gz", "NinetyNinePointFive.nii.gz", 0.5),
        ("OneHundred.nii.gz", "Zero.nii.gz", 100.0),
        (
            "OneHundred.nii.gz",
            "OneHundredwithGaussianNoiseSigmaOne.nii.gz",
            1.0003711823974208,
        ),
    ],
)
def test_perror(ref_fn, test_fn, expected_output):
    ref_fp = TEST_DATA_DIR / "perror" / ref_fn
    test_fp = TEST_DATA_DIR / "perror" / test_fn
    assert perror(ref_fp, test_fp) == expected_output


def test_check_overlay_args_missing_range():
    test_args = argparse.Namespace(
        ov=["a.nii", "b.nii", "c.nii"],
        ovr=[[1, 10], [2, 8]],
        ovc=["blue", "yellow", "green"],
        ova=[60, 50, 40],
    )

    with pytest.raises(
        RuntimeError,
        match="Each overlay must be accompanied by an intensity range using the -ovr argument",
    ):
        _ = rgboverlay.check_overlay_args(test_args)


def test_check_overlay_args_missing_colourmap():
    test_args = argparse.Namespace(
        ov=["a.nii", "b.nii", "c.nii"],
        ovr=[[1, 10], [2, 8], [5, 6]],
        ovc=["blue", "yellow"],
        ova=[60, 50, 40],
    )

    with pytest.raises(
        RuntimeError,
        match="Each overlay must be accompanied by a colour map using the -ovc argument",
    ):
        _ = rgboverlay.check_overlay_args(test_args)


def test_check_overlay_args_missing_opacity():
    test_args = argparse.Namespace(
        ov=["a.nii", "b.nii", "c.nii"],
        ovr=[[1, 10], [2, 8], [5, 6]],
        ovc=["blue", "yellow", "green"],
        ova=[60, 50],
    )

    with pytest.raises(
        RuntimeError,
        match="Each overlay must be accompanied by an opacity using the -ova argument",
    ):
        _ = rgboverlay.check_overlay_args(test_args)


def test_check_overlay_args_success():
    test_args = argparse.Namespace(
        ov=["a.nii", "b.nii", "c.nii"],
        ovr=[[1, 10], [2, 8], [5, 6]],
        ovc=["blue", "yellow", "red"],
        ova=[60, 50, 40],
    )

    try:
        rgboverlay.check_overlay_args(test_args)
    except RuntimeError:
        assert False


def test_check_nifti_datatype_typeerror():
    img = nib.nifti1.Nifti1Image(np.zeros([5, 5, 5]), np.eye(4))
    img.header["datatype"] = 32  # i.e. complex data, which is not allowed
    with pytest.raises(
        TypeError,
        match="NIfTI datatype must be 1, 2, 4, 8, 16, 64, 256, 512, 768, 1024, 1280 or 1536",
    ):
        _ = rgboverlay.check_nifti_datatype(img)


def test_check_nifti_datatype_success():
    img = nib.nifti1.Nifti1Image(np.zeros([5, 5, 5]), np.eye(4))
    img.header["datatype"] = 1  # allowed

    try:
        rgboverlay.check_nifti_datatype(img)
    except TypeError:
        assert False


def test_check_shape_and_orientation():
    affine_1 = np.eye(4)
    affine_2 = 2 * np.eye(4)
    affine_3 = np.eye(4)
    affine_3[0, 0] = 1 + 1e-4

    nii_obj_1 = nib.nifti1.Nifti1Image(np.ones((32, 32, 16)), affine_1)
    nii_obj_2 = nib.nifti1.Nifti1Image(np.ones((32, 32, 16)), affine_2)
    nii_obj_3 = nib.nifti1.Nifti1Image(np.ones((32, 32, 18)), affine_1)
    nii_obj_4 = nib.nifti1.Nifti1Image(np.ones((32, 32, 16)), affine_3)

    # Identical affine and matrix size
    assert rgboverlay.check_shape_and_orientation(nii_obj_1, nii_obj_1)
    # Different affine
    assert not rgboverlay.check_shape_and_orientation(nii_obj_1, nii_obj_2)
    # Different matrix size
    assert not rgboverlay.check_shape_and_orientation(nii_obj_1, nii_obj_3)
    # Different but within tolerance 1E-4
    assert rgboverlay.check_shape_and_orientation(nii_obj_1, nii_obj_4)


def test_create_rgb_mask():
    test_array = np.array([1, 2, 3, 4, 5])
    lo = 2

    mask = np.array([0, 0, 1, 1, 1])
    expected_output = np.stack((mask, mask, mask), axis=-1)
    assert np.allclose(rgboverlay.create_rgb_mask(test_array, lo), expected_output)


@pytest.mark.parametrize(
    "lut_name",
    ["red-yellow", "blue-lightblue", "red", "blue", "green", "yellow", "pink", "cool"],
)
def test_colourmap_lut(lut_name):
    array = rgboverlay.colourmap_lut(lut_name)

    assert array.shape == (256, 3)
    assert np.amax(array) <= 255
    assert np.amin(array) >= 0


def test_colourmap_error():
    # Unknown colour map
    with pytest.raises(ValueError):
        rgboverlay.colourmap_lut("hot")


def test_convert_to_4d_rgb_gray():
    test_im = np.array([[[np.nan, 2], [5, 6]], [[8, 9], [10, 11]]])
    gray_lut = rgboverlay.colourmap_lut("gray")

    im_4d_rgb = rgboverlay.convert_to_4d_rgb(test_im, 3, 10, gray_lut)

    assert im_4d_rgb.dtype == np.uint8
    assert np.min(im_4d_rgb) == 0
    assert np.max(im_4d_rgb) == 255
    assert im_4d_rgb.shape == test_im.shape + (3,)

    assert np.allclose(
        im_4d_rgb[:, :, :, 0], [[[0, 0], [73, 109]], [[182, 219], [255, 255]]]
    )
    assert np.allclose(
        im_4d_rgb[:, :, :, 1], [[[0, 0], [73, 109]], [[182, 219], [255, 255]]]
    )
    assert np.allclose(
        im_4d_rgb[:, :, :, 2], [[[0, 0], [73, 109]], [[182, 219], [255, 255]]]
    )


def test_convert_to_4d_rgb_gray_range_error():
    test_im = np.array([[[np.nan, 2], [5, 6]], [[8, 9], [10, 11]]])
    gray_lut = rgboverlay.colourmap_lut("gray")

    with pytest.raises(
        RuntimeError,
        match="The intensity range in the image after clipping with LO HI values is zero",
    ):
        _ = rgboverlay.convert_to_4d_rgb(test_im, 100, 150, gray_lut)


def test_convert_to_4d_rgb_red_yellow():
    test_im = np.array([[[np.nan, 2], [5, 6]], [[8, 9], [10, 11]]])
    grey_lut = rgboverlay.colourmap_lut("red-yellow")

    im_4d_rgb = rgboverlay.convert_to_4d_rgb(test_im, 3, 10, grey_lut)

    assert im_4d_rgb.dtype == np.uint8
    assert np.min(im_4d_rgb) == 0
    assert np.max(im_4d_rgb) == 255
    assert im_4d_rgb.shape == test_im.shape + (3,)
    assert np.allclose(
        im_4d_rgb[:, :, :, 0], [[[255, 255], [255, 255]], [[255, 255], [255, 255]]]
    )
    assert np.allclose(
        im_4d_rgb[:, :, :, 1], [[[0, 0], [73, 109]], [[182, 219], [255, 255]]]
    )
    assert np.allclose(im_4d_rgb[:, :, :, 2], [[[0, 0], [0, 0]], [[0, 0], [0, 0]]])


def test_blend():
    base_im = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    map_im = np.array([1, 1, 1, 5, 6, 5, 6, 2, 2, 2])
    mask = map_im > 3
    alpha = 0.5

    # possible to work out be hand
    expected_out = np.array([1, 2, 3, 4, 5, 5, 6, 8, 9, 10])

    test_out = rgboverlay.blend(base_im, map_im, mask, alpha)

    assert np.allclose(test_out, expected_out)


def test_cast2rgb_error_wrong_dimensions():
    # Incorrect number of dimensions
    with pytest.raises(ValueError):
        rgboverlay.cast2rgb(np.zeros([10, 10]))


def test_cast2rgb_error_wrong_4th_dim():
    # Incorrect shape of 4th dimension (should be 3)
    with pytest.raises(ValueError):
        rgboverlay.cast2rgb(np.zeros([10, 10, 10, 10]))


def test_cast2rgb():
    a = rgboverlay.cast2rgb(np.ones([5, 5, 5, 3], dtype=np.uint8))
    assert a.shape == (5, 5, 5)
    assert a.dtype == [("R", "u1"), ("G", "u1"), ("B", "u1")]


def test_prints_help_1(script_runner):
    result = script_runner.run(SCRIPT_NAME)
    assert result.success
    assert result.stdout.startswith(SCRIPT_USAGE)


def test_prints_help_2(script_runner):
    result = script_runner.run([SCRIPT_NAME, "-h"])
    assert result.success
    assert result.stdout.startswith(SCRIPT_USAGE)


def test_prints_help_for_invalid_option(script_runner):
    result = script_runner.run([SCRIPT_NAME, "-!"])
    assert not result.success
    assert result.stderr.startswith(SCRIPT_USAGE)


def test_prints_version(script_runner):
    result = script_runner.run([SCRIPT_NAME, "--version"])
    assert result.success
    expected_version_output = SCRIPT_NAME + " " + __version__ + "\n"
    assert result.stdout == expected_version_output


def test_rgboverlay_error_overlay_args(script_runner, tmp_path):
    input_test_data_dp = TEST_DATA_DIR / "input"

    base_fp = input_test_data_dp / "base.nii.gz"
    map_1_fp = input_test_data_dp / "map_1.nii.gz"
    map_2_fp = input_test_data_dp / "map_2.nii.gz"
    output_fp = tmp_path / "output.nii.gz"

    result = script_runner.run(
        [
            SCRIPT_NAME,
            "-b",
            base_fp,
            "-br",
            0,
            4095,
            "-ov",
            map_1_fp,
            "-ovr",
            "1",
            "5",
            "-ovc",
            "red",
            "-ova",
            60,
            "-ov",
            map_2_fp,
            "-ovr",
            "1",
            "5",
            "-ovc",
            "red",
            "-out",
            output_fp,
        ]
    )
    assert not result.success
    assert result.stderr.startswith(
        "Error: Each overlay must be accompanied by an opacity using the -ova argument\n"
    )


def test_rgboverlay_error_missing_base(script_runner, tmp_path):
    input_test_data_dp = TEST_DATA_DIR / "input"

    base_fp = input_test_data_dp / "base_not_found.nii.gz"
    map_1_fp = input_test_data_dp / "map_1.nii.gz"
    output_fp = tmp_path / "output.nii.gz"

    result = script_runner.run(
        [
            SCRIPT_NAME,
            "-b",
            base_fp,
            "-br",
            0,
            4095,
            "-ov",
            map_1_fp,
            "-ovr",
            "1",
            "5",
            "-ovc",
            "red",
            "-ova",
            60,
            "-out",
            output_fp,
        ]
    )
    assert not result.success
    assert result.stderr.startswith("Error: unable to load")


def test_rgboverlay_error_incorrect_base_nifti_datatype(script_runner, tmp_path):
    base_fp = tmp_path / "base_wrong_type.nii.gz"
    img = nib.nifti1.Nifti1Image(np.zeros([5, 5, 5]), np.eye(4))
    img.header["datatype"] = 32  # i.e. complex data, which is not allowed
    img.to_filename(base_fp)

    input_test_data_dp = TEST_DATA_DIR / "input"

    map_1_fp = input_test_data_dp / "map_1.nii.gz"
    output_fp = tmp_path / "output.nii.gz"

    result = script_runner.run(
        [
            SCRIPT_NAME,
            "-b",
            base_fp,
            "-br",
            0,
            4095,
            "-ov",
            map_1_fp,
            "-ovr",
            "1",
            "5",
            "-ovc",
            "red",
            "-ova",
            60,
            "-out",
            output_fp,
        ]
    )
    assert not result.success
    assert result.stderr.startswith(
        "Error: NIfTI datatype must be 1, 2, 4, 8, 16, 64, 256, 512, 768, 1024, 1280 or 1536"
    )


def test_rgboverlay_error_missing_overlay(script_runner, tmp_path):
    input_test_data_dp = TEST_DATA_DIR / "input"

    base_fp = input_test_data_dp / "base.nii.gz"
    map_1_fp = input_test_data_dp / "map_1_not_found.nii.gz"
    output_fp = tmp_path / "output.nii.gz"

    result = script_runner.run(
        [
            SCRIPT_NAME,
            "-b",
            base_fp,
            "-br",
            0,
            4095,
            "-ov",
            map_1_fp,
            "-ovr",
            "1",
            "5",
            "-ovc",
            "red",
            "-ova",
            60,
            "-out",
            output_fp,
        ]
    )
    assert not result.success
    assert result.stderr.startswith("Error: unable to load")


def test_rgboverlay_error_incorrect_overlay_nifti_datatype(script_runner, tmp_path):
    map_1_fp = tmp_path / "overlay_wrong_type.nii.gz"
    img = nib.nifti1.Nifti1Image(np.zeros([5, 5, 5]), np.eye(4))
    img.header["datatype"] = 32  # i.e. complex data, which is not allowed
    img.to_filename(map_1_fp)

    input_test_data_dp = TEST_DATA_DIR / "input"
    base_fp = input_test_data_dp / "base.nii.gz"
    output_fp = tmp_path / "output.nii.gz"

    result = script_runner.run(
        [
            SCRIPT_NAME,
            "-b",
            base_fp,
            "-br",
            0,
            4095,
            "-ov",
            map_1_fp,
            "-ovr",
            "1",
            "5",
            "-ovc",
            "red",
            "-ova",
            60,
            "-out",
            output_fp,
        ]
    )
    assert not result.success
    assert result.stderr.startswith(
        "Error: NIfTI datatype must be 1, 2, 4, 8, 16, 64, 256, 512, 768, 1024, 1280 or 1536"
    )


def test_rgboverlay_error_mismatched_affines(script_runner, tmp_path):
    input_test_data_dp = TEST_DATA_DIR / "input"

    base_fp = input_test_data_dp / "base.nii.gz"
    map_3_fp = input_test_data_dp / "map_3.nii.gz"
    output_fp = tmp_path / "output.nii.gz"

    result = script_runner.run(
        [
            SCRIPT_NAME,
            "-b",
            base_fp,
            "-br",
            0,
            4095,
            "-ov",
            map_3_fp,
            "-ovr",
            "1",
            "5",
            "-ovc",
            "red",
            "-ova",
            60,
            "-out",
            output_fp,
        ]
    )
    assert not result.success
    assert result.stderr.startswith(
        "Error: Base and overlay NIfTIs have mismatched geometry\n"
    )


def test_rgboverlay_base_range_error(script_runner, tmp_path):
    input_test_data_dp = TEST_DATA_DIR / "input"

    base_fp = input_test_data_dp / "base.nii.gz"
    map_1_fp = input_test_data_dp / "map_1.nii.gz"
    map_2_fp = input_test_data_dp / "map_2.nii.gz"
    output_fp = tmp_path / "output.nii.gz"

    result = script_runner.run(
        [
            SCRIPT_NAME,
            "-b",
            base_fp,
            "-br",
            5000,
            10000,
            "-ov",
            map_1_fp,
            "-ovr",
            "1",
            "5",
            "-ovc",
            "red-yellow",
            "-ova",
            60,
            "-ov",
            map_2_fp,
            "-ovr",
            "0.0011",
            "0.005",
            "-ovc",
            "blue-lightblue",
            "-ova",
            40,
            "-out",
            output_fp,
        ]
    )
    assert not result.success
    assert result.stderr.startswith(
        "Error: The intensity range in the image "
        "after clipping with LO HI values is zero, "
        "check -br values are appropriate given "
        "the range of voxel values in the base "
        "image\n"
    )


def test_rgboverlay_overlay_range_error(script_runner, tmp_path):
    input_test_data_dp = TEST_DATA_DIR / "input"

    base_fp = input_test_data_dp / "base.nii.gz"
    map_1_fp = input_test_data_dp / "map_1.nii.gz"
    map_2_fp = input_test_data_dp / "map_2.nii.gz"
    output_fp = tmp_path / "output.nii.gz"

    result = script_runner.run(
        [
            SCRIPT_NAME,
            "-b",
            base_fp,
            "-br",
            0,
            4000,
            "-ov",
            map_1_fp,
            "-ovr",
            "1",
            "5",
            "-ovc",
            "red-yellow",
            "-ova",
            60,
            "-ov",
            map_2_fp,
            "-ovr",
            "10",
            "20",
            "-ovc",
            "blue-lightblue",
            "-ova",
            40,
            "-out",
            output_fp,
        ]
    )
    assert not result.success
    assert result.stderr.startswith(
        "Error: The intensity range in the image "
        "after clipping with LO HI values is zero, "
        "check -ovr values are appropriate given "
        "the range of voxel values in the overlay "
        "image\n"
    )


def test_rgboverlay(script_runner, tmp_path):
    pthresh = 1.0

    input_test_data_dp = TEST_DATA_DIR / "input"
    output_test_data_dp = TEST_DATA_DIR / "output"

    base_fp = input_test_data_dp / "base.nii.gz"
    map_1_fp = input_test_data_dp / "map_1.nii.gz"
    map_2_fp = input_test_data_dp / "map_2.nii.gz"
    output_fp = tmp_path / "output.nii.gz"
    ref_output_fp = output_test_data_dp / "combined.nii.gz"

    result = script_runner.run(
        [
            SCRIPT_NAME,
            "-b",
            base_fp,
            "-br",
            0,
            4000,
            "-ov",
            map_1_fp,
            "-ovr",
            "1",
            "5",
            "-ovc",
            "red-yellow",
            "-ova",
            60,
            "-ov",
            map_2_fp,
            "-ovr",
            "0.0011",
            "0.005",
            "-ovc",
            "blue-lightblue",
            "-ova",
            40,
            "-out",
            output_fp,
        ]
    )
    assert result.success
    assert perror(ref_output_fp, output_fp) < pthresh
