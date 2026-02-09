
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

from utils import (get_bbox_from_mask, resize_image_and_bbox, 
                   compute_dice, compute_iou, save_results_to_csv)



@torch.no_grad()
def evaluate_baseline(medsam_model, data_loader, device, save_path=None):

    medsam_model.eval()
    results_list = []
    
    print("Evaluating MedSAM Baseline...")
    
    for batch in tqdm(data_loader, desc="Testing"):
        images = batch['image'].to(device)
        masks = batch['mask'].numpy()
        bboxes = batch['bbox'].to(device)
        
        img_embeddings = medsam_model.image_encoder(images)
        
        # Get prompt embeddings from bbox
        sparse_emb, dense_emb = medsam_model.prompt_encoder(
            points=None,
            boxes=bboxes[:, None, :],
            masks=None
        )
        
        # Predict masks
        low_res_logits, _ = medsam_model.mask_decoder(
            image_embeddings=img_embeddings,
            image_pe=medsam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        
        # Upsample to 1024
        low_res_pred = torch.sigmoid(low_res_logits)
        preds_1024 = torch.nn.functional.interpolate(
            low_res_pred,
            size=(1024, 1024),
            mode="bilinear",
            align_corners=False,
        )
        
        # Evaluate each sample
        for i in range(len(masks)):
            pred_1024 = preds_1024[i, 0].cpu().numpy() > 0.5
            gt_mask = masks[i]
            # print(f"\n[DEBUG] Sample {i}:")
            # print(f"  gt_mask original shape: {gt_mask.shape}")
            if gt_mask.ndim == 3:
                gt_mask = gt_mask.squeeze()
                # print(f"  gt_mask after squeeze: {gt_mask.shape}")
            
            # Resize prediction to original size
            H, W = batch['original_size'][0][i].item(), batch['original_size'][1][i].item()
            # print(f"  Target size (H, W): ({H}, {W})")
            # print(f"  pred_1024 shape: {pred_1024.shape}")
            from skimage import transform
            pred_resized = transform.resize(
                pred_1024.astype(float),
                (H, W),
                order=0,
                preserve_range=True,
                anti_aliasing=False
            ) > 0.5
            # print(f"  pred_resized shape: {pred_resized.shape}")
            # print(f"  gt_mask final shape: {gt_mask.shape}")
            
            gt_mask_resized = transform.resize(
                gt_mask.astype(float),
                (H, W),
                order=0,
                preserve_range=True,
                anti_aliasing=False
            ) > 0.5
    
            # Compute metrics
            dice = compute_dice(pred_resized, gt_mask_resized)
            iou = compute_iou(pred_resized, gt_mask_resized)
            
            # Store result
            result = {
                'dice': dice,
                'iou': iou,
            }
            
            # Add dataset-specific info
            if 'image_id' in batch:
                result['image_id'] = batch['image_id'][i]
                result['birads'] = batch['birads'][i]
            elif 'patient_id' in batch:
                result['patient_id'] = batch['patient_id'][i]
                result['t_stage'] = batch['t_stage'][i]
                result['histology'] = batch['histology'][i]
            
            results_list.append(result)
    
    # Save results
    if save_path:
        save_results_to_csv(results_list, save_path)
    
    mean_dice = np.mean([r['dice'] for r in results_list])
    return results_list, mean_dice


# ============================================================
# Text-Guided Evaluation
# ============================================================
@torch.no_grad()
def evaluate_text_guided(medsam_model, fusion_model, data_loader, device, save_path=None):

    medsam_model.eval()
    fusion_model.eval()
    results_list = []
    
    print("Evaluating Text-Guided MedSAM...")
    
    for batch in tqdm(data_loader, desc="Testing"):
        images = batch['image'].to(device)
        masks = batch['mask'].numpy()
        bboxes = batch['bbox'].to(device)
        text_embeds = batch['text_embed'].to(device)
        
        # Get image embeddings
        img_embeddings = medsam_model.image_encoder(images)
        
        # Apply text-guided fusion
        enhanced_embeddings = fusion_model(img_embeddings, text_embeds)
        
        # Get prompt embeddings
        sparse_emb, dense_emb = medsam_model.prompt_encoder(
            points=None,
            boxes=bboxes[:, None, :],
            masks=None
        )
        
        # Predict masks
        low_res_logits, _ = medsam_model.mask_decoder(
            image_embeddings=enhanced_embeddings,
            image_pe=medsam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        
        # Upsample
        low_res_pred = torch.sigmoid(low_res_logits)
        preds_1024 = torch.nn.functional.interpolate(
            low_res_pred,
            size=(1024, 1024),
            mode="bilinear",
            align_corners=False,
        )
        
        # Evaluate each sample
        for i in range(len(masks)):
            pred_1024 = preds_1024[i, 0].cpu().numpy() > 0.5
            gt_mask = masks[i]

            if gt_mask.ndim == 3:
                gt_mask = gt_mask.squeeze()
            
            # Resize to original size
            H, W = batch['original_size'][0][i].item(), batch['original_size'][1][i].item()
            from skimage import transform
            pred_resized = transform.resize(
                pred_1024.astype(float),
                (H, W),
                order=0,
                preserve_range=True,
                anti_aliasing=False
            ) > 0.
            gt_mask_resized = transform.resize(
                gt_mask.astype(float),
                (H, W),
                order=0,
                preserve_range=True,
                anti_aliasing=False
            ) > 0.5
            
            # Compute metrics
            dice = compute_dice(pred_resized, gt_mask_resized)
            iou = compute_iou(pred_resized, gt_mask_resized)
            
            # Store result
            result = {
                'dice': dice,
                'iou': iou,
            }
            
            # Add dataset-specific info
            if 'image_id' in batch:
                result['image_id'] = batch['image_id'][i]
                result['birads'] = batch['birads'][i]
            elif 'patient_id' in batch:
                result['patient_id'] = batch['patient_id'][i]
                result['t_stage'] = batch['t_stage'][i]
                result['histology'] = batch['histology'][i]
            
            results_list.append(result)
    
    # Save results
    if save_path:
        save_results_to_csv(results_list, save_path)
    
    mean_dice = np.mean([r['dice'] for r in results_list])
    return results_list, mean_dice


# ============================================================
# Counterfactual Experiment (NSCLC only)
# ============================================================
@torch.no_grad()
def evaluate_counterfactual(medsam_model, fusion_model, data_loader, device,
                           text_embeddings_dict, 
                           modify_t_stage=False, modify_histology=False,
                           save_path=None):

    medsam_model.eval()
    fusion_model.eval()
    results_list = []
    
    experiment_name = []
    if modify_t_stage:
        experiment_name.append("wrong T-stage")
    if modify_histology:
        experiment_name.append("wrong histology")
    
    if not experiment_name:
        experiment_name = ["correct prompts"]
    
    print(f"Counterfactual Experiment: {', '.join(experiment_name)}")
    
    for batch in tqdm(data_loader, desc="Testing"):
        images = batch['image'].to(device)
        masks = batch['mask'].numpy()
        bboxes = batch['bbox'].to(device)
        t_stages = batch['t_stage']
        histologies = batch['histology']
        
        # Get counterfactual text embeddings
        batch_text_embeds = []
        for i in range(len(t_stages)):
            t_stage = t_stages[i]
            histology = histologies[i]
            
            # Get T-stage embedding (correct or wrong)
            if modify_t_stage and t_stage in text_embeddings_dict['t_stage_cf_map']:
                wrong_t_stage = text_embeddings_dict['t_stage_cf_map'][t_stage]
                t_embed = text_embeddings_dict['t_stage'][wrong_t_stage]
            else:
                t_embed = text_embeddings_dict['t_stage'][t_stage]
            
            # Get histology embedding (correct or wrong)
            if modify_histology and histology in text_embeddings_dict['histology_cf_map']:
                wrong_histology = text_embeddings_dict['histology_cf_map'][histology]
                h_embed = text_embeddings_dict['histology'][wrong_histology]
            else:
                h_embed = text_embeddings_dict['histology'][histology]
            
            # Combine
            combined_embed = (t_embed + h_embed) / 2
            batch_text_embeds.append(combined_embed)
        
        text_embeds = torch.stack(batch_text_embeds).to(device)
        
        # Get image embeddings
        img_embeddings = medsam_model.image_encoder(images)
        
        # Apply fusion
        enhanced_embeddings = fusion_model(img_embeddings, text_embeds)
        
        # Get prompt embeddings
        sparse_emb, dense_emb = medsam_model.prompt_encoder(
            points=None,
            boxes=bboxes[:, None, :],
            masks=None
        )
        
        # Predict
        low_res_logits, _ = medsam_model.mask_decoder(
            image_embeddings=enhanced_embeddings,
            image_pe=medsam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        
        # Upsample
        low_res_pred = torch.sigmoid(low_res_logits)
        preds_1024 = torch.nn.functional.interpolate(
            low_res_pred,
            size=(1024, 1024),
            mode="bilinear",
            align_corners=False,
        )
        
        # Evaluate
        for i in range(len(masks)):
            pred_1024 = preds_1024[i, 0].cpu().numpy() > 0.5
            gt_mask = masks[i]
            if gt_mask.ndim == 3:
                gt_mask = gt_mask.squeeze()
            
            H, W = batch['original_size'][0][i].item(), batch['original_size'][1][i].item()
            from skimage import transform
            pred_resized = transform.resize(
                pred_1024.astype(float),
                (H, W),
                order=0,
                preserve_range=True,
                anti_aliasing=False
            ) > 0.5

            gt_mask_resized = transform.resize(
                gt_mask.astype(float),
                (H, W),
                order=0,
                preserve_range=True,
                anti_aliasing=False
            ) > 0.5
            
            dice = compute_dice(pred_resized, gt_mask_resized)
            iou = compute_iou(pred_resized, gt_mask_resized)
            
            result = {
                'dice': dice,
                'iou': iou,
                'patient_id': batch['patient_id'][i],
                't_stage': t_stages[i],
                'histology': histologies[i],
                'modified_t_stage': modify_t_stage,
                'modified_histology': modify_histology
            }
            
            results_list.append(result)
    
    if save_path:
        save_results_to_csv(results_list, save_path)
    
    mean_dice = np.mean([r['dice'] for r in results_list])
    return results_list, mean_dice


@torch.no_grad()
def evaluate_bus_counterfactual(medsam_model, fusion_model, data_loader, device,
                                birads_embeddings, counterfactual_map,
                                use_wrong_birads=True, save_path=None):

    medsam_model.eval()
    fusion_model.eval()
    results_list = []
    
    experiment_name = "wrong BI-RADS" if use_wrong_birads else "correct BI-RADS"
    print(f"BUS Counterfactual Experiment: {experiment_name}")
    
    for batch in tqdm(data_loader, desc="Testing"):
        images = batch['image'].to(device)
        masks = batch['mask'].numpy()
        bboxes = batch['bbox'].to(device)
        birads_list = batch['birads']
        
        # Get counterfactual text embeddings
        # batch_text_embeds = []
        # for birads in birads_list:
        #     if use_wrong_birads and birads in counterfactual_map:
        #         wrong_birads = counterfactual_map[birads]
        #         text_embed = birads_embeddings[wrong_birads]
        #     else:
        #         text_embed = birads_embeddings[birads]
            
        #     batch_text_embeds.append(text_embed)
        
        batch_text_embeds = []
        for birads in birads_list:

            if isinstance(birads, torch.Tensor):
                birads_int = int(birads.item())
            else:
                birads_int = int(birads)


            if use_wrong_birads and birads_int in counterfactual_map:
                wrong_birads_int = int(counterfactual_map[birads_int])
                text_embed = birads_embeddings[wrong_birads_int]
            else:
                text_embed = birads_embeddings[birads_int]

            batch_text_embeds.append(text_embed)
        
        text_embeds = torch.stack(batch_text_embeds).to(device)
        
        if 'image_embedding' in batch:
            img_embeddings = batch['image_embedding'].to(device)
        else:
            img_embeddings = medsam_model.image_encoder(images)
        
        # Apply fusion
        enhanced_embeddings = fusion_model(img_embeddings, text_embeds)
        
        # Get prompt embeddings
        sparse_emb, dense_emb = medsam_model.prompt_encoder(
            points=None,
            boxes=bboxes[:, None, :],
            masks=None
        )
        
        # Predict
        low_res_logits, _ = medsam_model.mask_decoder(
            image_embeddings=enhanced_embeddings,
            image_pe=medsam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        
        # Upsample
        low_res_pred = torch.sigmoid(low_res_logits)
        preds_1024 = torch.nn.functional.interpolate(
            low_res_pred,
            size=(1024, 1024),
            mode="bilinear",
            align_corners=False,
        )
        
        # Evaluate
        for i in range(len(masks)):
            pred_1024 = preds_1024[i, 0].cpu().numpy() > 0.5
            gt_mask = masks[i]
            
            if gt_mask.ndim == 3:
                gt_mask = gt_mask.squeeze()
            
            H, W = batch['original_size'][0][i].item(), batch['original_size'][1][i].item()
            from skimage import transform
            
            pred_resized = transform.resize(
                pred_1024.astype(float),
                (H, W),
                order=0,
                preserve_range=True,
                anti_aliasing=False
            ) > 0.5
            
            gt_mask_resized = transform.resize(
                gt_mask.astype(float),
                (H, W),
                order=0,
                preserve_range=True,
                anti_aliasing=False
            ) > 0.5
            
            dice = compute_dice(pred_resized, gt_mask_resized)
            iou = compute_iou(pred_resized, gt_mask_resized)
            
            result = {
                'dice': dice,
                'iou': iou,
                'image_id': batch['image_id'][i],
                'birads': birads_list[i],
                'used_wrong_birads': use_wrong_birads
            }
            
            results_list.append(result)
    
    if save_path:
        save_results_to_csv(results_list, save_path)
    
    mean_dice = np.mean([r['dice'] for r in results_list])
    return results_list, mean_dice

# ============================================================
# Compare Results
# ============================================================
def compare_results(baseline_results, text_guided_results, dataset_name="Dataset"):

    baseline_dice = [r['dice'] for r in baseline_results]
    text_guided_dice = [r['dice'] for r in text_guided_results]
    
    # print(f"\n{'='*70}")
    print(f"COMPARISON: {dataset_name}")
    # print(f"{'='*70}")
    print(f"Baseline MedSAM:")
    print(f"  Mean Dice: {np.mean(baseline_dice):.4f} ± {np.std(baseline_dice):.4f}")
    print(f"\nText-Guided MedSAM:")
    print(f"  Mean Dice: {np.mean(text_guided_dice):.4f} ± {np.std(text_guided_dice):.4f}")
    
    improvement = np.mean(text_guided_dice) - np.mean(baseline_dice)
    print(f"\nImprovement: {improvement:.4f} ({improvement/np.mean(baseline_dice)*100:.2f}%)")
    # print(f"{'='*70}\n")