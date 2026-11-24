# mask_3d
*Segments masks in 3D*

## Invoke
Inside the folder with your input data, run:
```shell
pyhim -C mask_3d
```

![segmentation](../../../_static/from_tuto/segmentation.png)

## Inputs

|Name shape|Quantity|Mandatory|Description|
|---|---|---|---|
|parameters.json|1|Yes|Parameter file.|
|<image_name>.tif|2..n|Yes|3D images|

## Outputs
|Name shape|Quantity|Description|
|---|---|---|
|*_3Dmasks.npy|2..n||


A 3D mask segmentation produces two outputs saved in the `segmentedObjects` folder:

```
scan_002_mask0_002_ROI_converted_decon_ch01.tif_3Dmasks.png
scan_002_mask0_002_ROI_converted_decon_ch01_3Dmasks.npy
```

The PNG file is a representation of the raw image and the segmented objects.

The NPY file is a 3D labeled numpy array containing the segmented objects. The file name is constructed using the original root filename with the tag `_3Dmasks`.

_Warning_: This mode operates in 3D, therefore the Startdist network provided **must be** in 3D.

## Separating touching objects

After StarDist prediction, `mask_3d` applies **only the first watershed pass**
of the implementation shared with `localize_3d`. It uses peaks in the foreground
distance transform as markers. The first-pass seed detection settings are
unchanged. The second pass, which subdivides elongated objects, is skipped
for masks to test whether a single pass provides sufficient separation.
`localize_3d` continues to use both passes. The resulting instance labels are saved directly,
without connected-component relabeling that would merge touching objects.

This procedure recomputes labels from the binary foreground: it does not
preserve StarDist IDs or use intensity differences to split objects. A contact
alone does not guarantee a split; the distance peaks determine the markers.
Peak searches use a `(10, 10, 25)` voxel footprint in the first step. These
settings are shared with the localization procedure and may need adaptation
for masks whose size differs substantially from localization spots.

## Relevant options
Most of the parameters are shared with ```mask_2d```, except for the following:

|Name|Option|Description|
|:-:|:-:|:-:|
|stardist_basename| | Folder containing 2D and 3D AI models|
|stardist_network3D| | Name of the 3D network|
