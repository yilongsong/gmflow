import torch
import numpy as np
from gmflow.gmflow import GMFlow



# seed = 326
# torch.manual_seed(seed)
# np.random.seed(seed)

# torch.backends.cudnn.benchmark = True

# checkpoint_dir = 'tmp'
# stage = 'chairs'
# image_size = [128, 128]
# padding_factor = 16
# max_flow = 400
# val_dataset = ['chair']
# with_speed_metric = False

# lr = 4e-4
# batch_size = 12
# num_workers = 4
# weight_decay = 1e-4
# grad_clip = 1.0
# num_steps = 100000
# seed = 326
# summary_freq = 100
# val_freq = 10000
# save_ckpt_freq = 10000
# save_latest_ckpt_freq = 1000

# resume = 'pretrained/gmflow_sintel=oc07dcb3.pth' #####

# no_resume_optimizer = False

strict_resume = False ###
num_scales = 1 ###
feature_channels = 128 ###
upsample_factor = 8 ###
num_transformer_layers = 6 ###
num_head = 1 ###
attention_type = 'swin' ###
ffn_dim_expansion = 4 ###

attn_splits_list = [2]
corr_radius_list = [-1]
prop_radius_list = [-1]
pred_bidir_flow = False

# gamma = 0.9

# eval = False
# save_eval_to_file = False
# evaluate_matched_unmatched = False

# inference_dir = 'demo/sintel_market_1' #####
# inference_size = None
# dir_paired_data = False ### Look into
# save_flo_flow = False
# fwd_bwd_consistency_check = False

# submission = False
# output_path = 'output/gmflow-norefine-sintel_market_1' #####
# save_vis_flow = False
# no_save_flo = False

local_rank = 0
# distributed = False
# launcher = 'none'
# gpu_ids = 0
# count_time = False

# def main():
#     torch.manual_seed(seed)
#     np.random.seed(seed)

#     torch.backends.cudnn.benchmark = True

#     distributed = False
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#     model = GMFlow(feature_channels=feature_channels,
#                    num_scales=num_scales,
#                    upsample_factor=upsample_factor,
#                    num_head=num_head,
#                    attention_type=attention_type,
#                    ffn_dim_expansion=ffn_dim_expansion,
#                    num_transformer_laers=num_transformer_layers,
#                    ).to(device)
    
#     if torch.cuda.device_count() > 1:
#         print('Use %d GPUs' % torch.cuda.device_count())
#         model = torch.nn.DataParallel(model)

#         model_without_ddp = model.module

#     else:
#         model_without_ddp = model

#     num_params = sum(p.numel() for p in model.parameters())
#     print('Number of params:', num_params)
    
#     optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=lr,
#                                   weight_decay=weight_decay)
    
#     start_epoch = 0
#     start_step = 0

#     print('Load checkpoint: %s' % resume)

#     loc = 'cuda:{}'.format(local_rank)
#     checkpoint = torch.load(resume, map_location=loc)
#     weights = checkpoint['model'] if 'model' in checkpoint else checkpoint

#     model_without_ddp.load_state_dict(weights, strict=strict_resume)

#     if 'optimizer' in checkpoint and 'step' in checkpoint and 'epoch' in checkpoint and not \
#                 no_resume_optimizer:
#         print('Load optimizer')
#         optimizer.load_state_dict(checkpoint['optimizer'])
#         start_epoch = checkpoint['epoch']
#         start_step = checkpoint['step']

#     print('start_epoch: %d, start_step: %d' % (start_epoch, start_step))

#     inference_on_dir(model_without_ddp,
#                          inference_dir=inference_dir,
#                          output_path=output_path,
#                          padding_factor=padding_factor,
#                          inference_size=inference_size,
#                          paired_data=dir_paired_data,
#                          save_flo_flow=save_flo_flow,
#                          attn_splits_list=attn_splits_list,
#                          corr_radius_list=corr_radius_list,
#                          prop_radius_list=prop_radius_list,
#                          pred_bidir_flow=pred_bidir_flow,
#                          fwd_bwd_consistency_check=fwd_bwd_consistency_check,
#                          )

from glob import glob
import h5py
from tqdm import tqdm
import numpy as np
import torch
from matplotlib import pyplot as plt

def visualize_optical_flow(flow):
    x, y = np.meshgrid(np.arange(0, 128, 8), np.arange(0, 128, 8))

    # Extract the u and v components of optical flow
    u = flow[:,:,0]
    v = flow[:,:,1]

    # Plot the optical flow using arrows
    plt.figure(figsize=(10, 5))
    plt.imshow(np.zeros((128, 128)), cmap='gray')  # Plot a blank image
    plt.quiver(x, y, u[::8, ::8], v[::8, ::8], color='red', angles='xy', scale_units='xy', scale=1)
    plt.title('Optical Flow Visualization')
    plt.xlabel('Horizontal')
    plt.ylabel('Vertical')
    plt.axis('equal')
    plt.show()

def get_gmflow_model(resume):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GMFlow(feature_channels=feature_channels,
                   num_scales=num_scales,
                   upsample_factor=upsample_factor,
                   num_head=num_head,
                   attention_type=attention_type,
                   ffn_dim_expansion=ffn_dim_expansion,
                   num_transformer_laers=num_transformer_layers,
                   ).to(device)
    
    '''
    Single GPU version
    '''
    # if torch.cuda.device_count() > 1:
    #     print('Use %d GPUs' % torch.cuda.device_count())
    #     model = torch.nn.DataParallel(model)
    #     model_without_ddp = model.module

    # else:
    #     model_without_ddp = model

    loc = 'cuda:{}'.format(local_rank)
    checkpoint = torch.load(resume, map_location=loc)
    weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(weights, strict=strict_resume)

    return model

def get_gmflow_flow(model, obs, next_obs):
    image1 = torch.from_numpy(obs).permute(2, 0, 1).float()
    image2 = torch.from_numpy(next_obs).permute(2, 0, 1).float()
    image1, image2 = image1[None].cuda(), image2[None].cuda()
    
    results_dict = model(image1, image2,
                             attn_splits_list=attn_splits_list,
                             corr_radius_list=corr_radius_list,
                             prop_radius_list=prop_radius_list,
                             pred_bidir_flow=pred_bidir_flow,
                             )
    
    flow = results_dict['flow_preds'][-1][0].permute(1,2,0).cpu().detach().numpy()

    return flow

def main():
    # datasets_path = "/users/ysong135/scratch/datasets/" # Oscar
    datasets_path = "/home/yilong/Documents/videopredictor/datasets/" # Local
    sequence_dirs = glob(f"{datasets_path}/**/*.hdf5", recursive=True)

    resume = '/home/yilong/Documents/videopredictor/flowdiffusion/gmflow/pretrained/gmflow_sintel-0c07dcb3.pth'

    flow_model = get_gmflow_model(resume)


    for seq_dir in sequence_dirs:
        print(f'Adding optical flow to {seq_dir}')
        task = seq_dir.split("/")[-2].replace('_', ' ')
        with h5py.File(seq_dir, 'r') as f:
            data = f['data']
            for demo in tqdm(data):
                next_obs = f['data'][demo]['next_obs']['sideview_image'][3:][::4]
                obs = f['data'][demo]['obs']['sideview_image'][::3][:len(next_obs)]

                for i in range(len(obs)):
                    flow = get_gmflow_flow(flow_model, obs[i], next_obs[i])
                    if i == len(obs)//2:
                        visualize_optical_flow(flow)


if __name__ == '__main__':
    main()