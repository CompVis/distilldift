import torch
from typing import Optional, Tuple

from pytorch3d.renderer.cameras import get_screen_to_ndc_transform
from pytorch3d.implicitron.tools.point_cloud_utils import _transform_points, PointsRasterizer, PointsRasterizationSettings

def get_frame_data(dataset, sequence_name, max_frames):
    """
    Get frame data from a dataset given a sequence name.

    Args:
        dataset (JsonIndexDataset): Dataset object.
        sequence_name (str): Sequence name.
        max_frames (int): Maximum number of frames.
    """
    sequence_entries = list(range(len(dataset)))
    sequence_entries = [i for i in sequence_entries if dataset.frame_annots[i]["frame_annotation"].sequence_name == sequence_name]
    if len(sequence_entries) == 0:
        return None
    if len(sequence_entries) > max_frames:
        sequence_entries = [sequence_entries[int(i * len(sequence_entries) / max_frames)] for i in range(max_frames)]

    sequence_dataset = torch.utils.data.Subset(dataset, sequence_entries)
    loader = torch.utils.data.DataLoader(
        sequence_dataset,
        batch_size=len(sequence_dataset),
        shuffle=False,
        num_workers=1,
        collate_fn=dataset.frame_data_type.collate,
    )
    frame_data = next(iter(loader))  # there's only one batch
    return frame_data

def pointcloud_to_depth(
    camera,
    point_cloud,
    render_size: Tuple[int, int],
    point_radius: float = 0.03,
    topk: int = 10,
    eps: float = 1e-2,
    bin_size: Optional[int] = None,
    **kwargs,
):
    """
    Render a point cloud to a depth map.

    Args:
        camera (Camera): Camera object.
        point_cloud (Tensor): Point cloud in world space (X, Y, Z).
        render_size (Tuple[int, int]): Render size.
        point_radius (float): Point radius.
        topk (int): Number of points per pixel.
        eps (float): Epsilon.
        bin_size (Optional[int]): Bin size.
    """
    # move to the camera coordinates; using identity cameras in the renderer
    point_cloud = _transform_points(camera, point_cloud, eps, **kwargs)
    camera_trivial = camera.clone()
    camera_trivial.R[:] = torch.eye(3)
    camera_trivial.T *= 0.0

    bin_size = (
        bin_size
        if bin_size is not None
        else (64 if int(max(render_size)) > 1024 else None)
    )
    rasterizer = PointsRasterizer(
        cameras=camera_trivial,
        raster_settings=PointsRasterizationSettings(
            image_size=render_size,
            radius=point_radius,
            points_per_pixel=topk,
            bin_size=bin_size,
        ),
    )

    fragments = rasterizer(point_cloud, **kwargs)
    return fragments.zbuf.max(dim=-1)[0]

def points_2d_to_3d(nz_indices, depth):
    """
    Convert 2D points to 3D points using depth map in (X, Y, Z) format (screen space).

    Args:
        nz_indices (Tensor): 2D points in screen space (X, Y).
        depth (Tensor): Depth map.
    """
    depth_values = depth[nz_indices[:, 1], nz_indices[:, 0]]
    return torch.stack((nz_indices[:, 0], nz_indices[:, 1], depth_values), dim=1) # X, Y, Z in screen space

def points_3d_to_2d(points_3d, image_size):
    """
    Convert 3D points to 2D points in screen space in (X, Y) format (screen space).

    Args:
        points_3d (Tensor): 3D points in screen space (Y, X, Z).
        image_size (Tuple[int, int]): Image size.
    """
    return points_3d[:, [1, 0]].long().clamp(0, image_size[0] - 1)

def screen_to_world(points_screen, camera, image_size):
    """
    Convert screen space points to world space (X, Y, Z).

    Args:
        points_screen (Tensor): Points (X, Y, Z) in screen space.
        camera (Camera): Camera object.
        image_size (Tuple[int, int]): Image size.
    """
    points_ndc = get_screen_to_ndc_transform(camera, image_size=image_size, with_xyflip=True).transform_points(points_screen) # X, Y, Z in NDC
    points_world = camera.unproject_points(points_ndc, world_coordinates=True, from_ndc=True) # X, Y, Z in world space
    return points_world

def world_to_screen(points_world, camera, image_size):
    """
    Convert world space points to screen space (Y, X, Z).

    Args:
        points_world (Tensor): Points (X, Y, Z) in world space.
        camera (Camera): Camera object.
        image_size (Tuple[int, int]): Image size.
    """
    points_screen = camera.transform_points_screen(points_world, image_size=image_size) # Y, X, Z in screen space
    return points_screen
