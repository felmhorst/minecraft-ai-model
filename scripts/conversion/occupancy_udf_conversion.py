import torch
import scipy.ndimage


def occupancy_to_udf(occupancy: torch.Tensor, max_distance: float = 1.0) -> torch.Tensor:
    """
    Convert a binary occupancy field to an unsigned distance field (UDF/TDF).

    occupancy: [B, 1, D, H, W] tensor of 0/1 values
    max_distance: maximum distance to normalize/truncate
    returns: [B, 1, D, H, W] tensor of floats
    """
    udf_list = []
    occupancy_np = occupancy.cpu().numpy()  # convert to numpy for scipy

    for b in range(occupancy.shape[0]):
        occ = occupancy_np[b, 0]  # [D,H,W]
        # compute distance transform: distance to nearest occupied voxel
        dist = scipy.ndimage.distance_transform_edt(1 - occ)  # distance outside the object
        dist = torch.from_numpy(dist).float()
        # optional normalization / truncation
        dist = torch.clamp(dist, 0, max_distance) / max_distance
        udf_list.append(dist.unsqueeze(0))

    return torch.stack(udf_list, dim=0).to(occupancy.device)  # [B,1,D,H,W]


def udf_to_occupancy(udf: torch.Tensor, threshold: float = 0.5, max_distance: float = 1.0) -> torch.Tensor:
    """
    Convert a UDF/TDF tensor back to occupancy (0/1)

    udf: [B,1,D,H,W], values in [0,1]
    threshold: normalized threshold to decide occupancy
    max_distance: used to map back normalized values
    """
    # scale back to actual distance
    dist = udf * max_distance
    occupancy = (dist <= threshold * max_distance).float()
    return occupancy
