# 这个代码在服务器可以跑通
import medim
import torch
import torch.nn.functional as F
import numpy as np
def random_sample_next_click(prev_mask, gt_mask, method='random'):
    """
    Randomly sample one click from ground-truth mask and previous seg mask

    Arguements:
        prev_mask: (torch.Tensor) [H,W,D] previous mask that SAM-Med3D predict
        gt_mask: (torch.Tensor) [H,W,D] ground-truth mask for this image
    """
    def ensure_3D_data(roi_tensor):
        if roi_tensor.ndim != 3:
            roi_tensor = roi_tensor.squeeze()
        assert roi_tensor.ndim == 3, "Input tensor must be 3D"
        return roi_tensor

    prev_mask = ensure_3D_data(prev_mask)
    gt_mask = ensure_3D_data(gt_mask)

    prev_mask = prev_mask > 0
    true_masks = gt_mask > 0

    if not true_masks.any():
        raise ValueError("Cannot find true value in the ground-truth!")

    fn_masks = torch.logical_and(true_masks, torch.logical_not(prev_mask))
    fp_masks = torch.logical_and(torch.logical_not(true_masks), prev_mask)

    if method.lower() == 'random':
        to_point_mask = torch.logical_or(fn_masks, fp_masks)  # error region

        if not to_point_mask.any():
            all_points = torch.argwhere(true_masks)
            point = all_points[np.random.randint(len(all_points))]
            is_positive = True
        else:
            all_points = torch.argwhere(to_point_mask)
            point = all_points[np.random.randint(len(all_points))]
            is_positive = bool(fn_masks[point[0], point[1], point[2]])

        sampled_point = point.clone().detach().reshape(1, 1, 3)
        sampled_label = torch.tensor([[int(is_positive)]], dtype=torch.long)

        return sampled_point, sampled_label

    elif method.lower() == 'ritm':
        # Pad masks and compute EDT
        fn_mask_single = F.pad(fn_masks[None, None], (1, 1, 1, 1, 1, 1), "constant", value=0).to(torch.uint8)[0, 0]
        fp_mask_single = F.pad(fp_masks[None, None], (1, 1, 1, 1, 1, 1), "constant", value=0).to(torch.uint8)[0, 0]

        fn_mask_dt = torch.tensor(edt.edt(fn_mask_single.cpu().numpy(), black_border=True, parallel=4))[1:-1, 1:-1, 1:-1]
        fp_mask_dt = torch.tensor(edt.edt(fp_mask_single.cpu().numpy(), black_border=True, parallel=4))[1:-1, 1:-1, 1:-1]

        fn_max_dist = torch.max(fn_mask_dt)
        fp_max_dist = torch.max(fp_mask_dt)

        is_positive = fn_max_dist > fp_max_dist
        dt = fn_mask_dt if is_positive else fp_mask_dt
        max_dist = max(fn_max_dist, fp_max_dist)

        to_point_mask = (dt > (max_dist / 2.0))
        all_points = torch.argwhere(to_point_mask)

        if len(all_points) == 0:
            # fallback: center of volume
            point = torch.tensor([gt_mask.shape[0] // 2, gt_mask.shape[1] // 2, gt_mask.shape[2] // 2])
            is_positive = False
        else:
            point = all_points[np.random.randint(len(all_points))]
            is_positive = bool(fn_masks[point[0], point[1], point[2]])

        sampled_point = point.clone().detach().reshape(1, 1, 3)
        sampled_label = torch.tensor([[int(is_positive)]], dtype=torch.long)

        return sampled_point, sampled_label

    else:
        raise ValueError(f"Unsupported method: {method}. Choose 'ritm' or 'random'.")


def sam_model_infer(model,
                    roi_image,
                    roi_gt=None,
                    prompt_generator=random_sample_next_click,
                    prev_low_res_mask=None,
                    num_clicks=1): # Added num_clicks for iterative prompting if desired
    '''
    Inference for SAM-Med3D, inputs prompt points with its labels (positive/negative for each points)
    '''
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if roi_gt is not None and (roi_gt == 0).all() and num_clicks > 0:
        # If GT is empty, and we need clicks, result is likely empty.
        # SAM might still predict something with a central click, but let's return empty.
        print("Warning: roi_gt is empty. Prediction will be empty.")
        return np.zeros_like(roi_image.cpu().numpy().squeeze()), None # Return None for low_res_mask

    with torch.no_grad():
        input_tensor = roi_image.to(device)
        image_embeddings = model.image_encoder(input_tensor) # image_embeddings.shape=torch.Size([B, 384, 8, 8, 8])

        points_coords, points_labels = torch.zeros(1, 0, 3).to(device), torch.zeros(1,
                                                                                    0).to(device)
        new_points_co, new_points_la = torch.Tensor([[[64, 64, 64]]]).to(device), torch.Tensor(
            [[1]]).to(torch.int64)
        
        current_prev_mask_for_click_generation = torch.zeros_like(roi_image, device=device)[:,0,...] # Start with empty prev_mask for click
        
        if prev_low_res_mask is None: # Initialize low_res_mask for the decoder
             prev_low_res_mask = torch.zeros(1, 1, roi_image.shape[2] // 4,
                                             roi_image.shape[3] // 4,
                                             roi_image.shape[4] // 4, device=device, dtype=torch.float) # prev_low_res_mask.shape=torch.Size([1, 1, 32, 32, 32])


        for _ in range(num_clicks):
            if roi_gt is not None:
                new_points_co, new_points_la = prompt_generator(
                    current_prev_mask_for_click_generation.squeeze(0).cpu(), # Expects HWD tensor
                    roi_gt[0, 0].cpu() # Expects HWD tensor
                )
                new_points_co, new_points_la = new_points_co.to(device), new_points_la.to(device)
            else: # No GT, default to a central positive click for the first click
                if points_coords.shape[1] == 0: # Only for the very first click if no GT
                    center_z = roi_image.shape[2] // 2
                    center_y = roi_image.shape[3] // 2
                    center_x = roi_image.shape[4] // 2
                    new_points_co = torch.tensor([[[center_x, center_y, center_z]]], device=device, dtype=torch.float) # X,Y,Z for SAM points
                    new_points_la = torch.tensor([[1]], device=device, dtype=torch.int64)
                else: # Subsequent clicks without GT are problematic, break or use last mask
                    print("Warning: No ground truth for subsequent click generation.")
                    break 
            
            points_coords = torch.cat([points_coords, new_points_co], dim=1) # points_coords.shape=torch.Size([1, 1, 3])
            points_labels = torch.cat([points_labels, new_points_la], dim=1) # points_labels.shape=torch.Size([1, 1])
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=[points_coords, points_labels],
                boxes=None,
                masks=prev_low_res_mask,
            )

            low_res_masks, _ = model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
            )
            # Update prev_low_res_mask for next iteration's prompt encoder input
            prev_low_res_mask = low_res_masks.detach() 

            # For click generation, use the upscaled version of the current prediction
            current_prev_mask_for_click_generation = F.interpolate(low_res_masks,
                                   size=roi_image.shape[-3:],
                                   mode='trilinear',
                                   align_corners=False)
            current_prev_mask_for_click_generation = torch.sigmoid(current_prev_mask_for_click_generation) > 0.5


        # Final high-resolution mask from the last low_res_masks
        final_masks_hr = F.interpolate(low_res_masks, # Use the final low_res_masks
                                       size=roi_image.shape[-3:],
                                       mode='trilinear',
                                       align_corners=False) # torch.Size([B, 1, 128, 128, 128])

    medsam_seg_prob = torch.sigmoid(final_masks_hr) 
    medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze() # medsam_seg_prob.shape=(B, 128, 128, 128)
    medsam_seg_mask = (medsam_seg_prob > 0.5).astype(np.uint8) # medsam_seg_mask.shape=(B, 128, 128, 128)

    return medsam_seg_mask, low_res_masks.detach()

device = "cuda:2"
prompt_generator = random_sample_next_click
model = medim.create_model("SAM-Med3D", pretrained=True, checkpoint_path="./sam_med3d_turbo.pth").to(device)
# 我有这两个数据集，mmwhs和pro128
dataset = "mmwhs" # pro128
if dataset == "mmwhs":
    roi_image = torch.rand(2,1,96,96,96).to(device)
elif dataset == "pro128":
    roi_image = torch.rand(2,1,32,128,128).to(device)
# B * 1 *  128 * 128 * 128, 所以我对输入进行了pad操作，但是输出的结果改如何还原呢？这是个问题
d_pad = 128 - roi_image.shape[2]
h_pad = 128 - roi_image.shape[3]
w_pad = 128 - roi_image.shape[4]

# F.pad format: (len_last_dim, len_2nd_last_dim, ...) -> (W_left, W_right, H_top, H_bottom, D_front, D_back)
# We pad at the end to keep coordinates valid
pad_params = (0, max(0, w_pad), 0, max(0, h_pad), 0, max(0, d_pad))

if sum(pad_params) > 0:
    roi_image_padded = F.pad(roi_image, pad_params, "constant", 0)
else:
    roi_image_padded = roi_image
sam_model_infer(model, roi_image_padded)





