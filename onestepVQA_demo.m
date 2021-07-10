clc
clear
close all

%% This is the sample code for 1stepVQA model.
% Yu, Xiangxu, et al. "Predicting the quality of compressed videos with pre-existing distortions." 
% arXiv preprint arXiv:2004.02943 (2020).
% https://github.com/xiangxuyu/1stepVQA

addpath('function')

tic

frameheight = 1080;
framewidth = 1920;

% sample reference video and test video from LIVE Wild Compressed Video Quality Database
% https://live.ece.utexas.edu/research/onestep/index.html
name_reference = 'A001.yuv';
name_compressed = 'A001-360p-37-1080p.yuv';

%% Compute features of reference video

% read frame number
filep            = dir(name_reference);
fileBytes        = filep.bytes;
fwidth           = 0.5;
fheight          = 0.5;
clear filep
Nframes      = fileBytes/(framewidth*frameheight*(1+2*fheight*fwidth));

for frame_ind = 1 : 1 : (Nframes - 1)
    
    fprintf('reference video name: %s;\n Current frame: %i\n', name_reference, frame_ind);
    
    % read current and next frames
    frameNow             = read_single_frame(name_reference, frame_ind, frameheight, framewidth);
    frameNext            = read_single_frame(name_reference, (frame_ind + 1), frameheight, framewidth);
    
    % compute spatial and temporal features
    feat_spatial_now     = onestepVQA_spatial(frameNow);
    feat_temporal_now    = onestepVQA_temporal(frameNow, frameNext);
    feat_reference(frame_ind, :) = [feat_spatial_now feat_temporal_now];
    
end

% average features of all frames
feat_reference_mean = mean(feat_reference);

%% Compute features of compressed video

% read frame number
filep            = dir(name_compressed);
fileBytes        = filep.bytes;
fwidth           = 0.5;
fheight          = 0.5;
clear filep
Nframes      = fileBytes/(framewidth*frameheight*(1+2*fheight*fwidth));

for frame_ind = 1 : 1 : (Nframes - 1)
    
    fprintf('Compressed video name: %s;\n Current frame: %i\n', name_compressed, frame_ind);
    
    %read current and next frames
    frameNow             = read_single_frame(name_compressed, frame_ind, frameheight, framewidth);
    frameNext            = read_single_frame(name_compressed, (frame_ind + 1), frameheight, framewidth);
    
    % compute spatial and temporal features
    feat_spatial_now     = onestepVQA_spatial(frameNow);
    feat_temporal_now    = onestepVQA_temporal(frameNow, frameNext);
    feat_compressed(frame_ind, :) = [feat_spatial_now feat_temporal_now];
    
end

% average features of all frames
feat_compressed_mean = mean(feat_compressed);

%% MAX operation, keep features with largest alpha of GGD parameters

% extract four alpha parameters of temporal features of reference video
alpha_reference = [feat_reference_mean(5) feat_reference_mean(7) feat_reference_mean(9) feat_reference_mean(11)];
% find the displacement direction (location) of max alpha
d_max = find(feat_reference_mean == max(alpha_reference));
% compute index of temporal features (2 scales)
feat_temporal_ind = [d_max d_max+1 d_max+8 d_max+9];

% final 1stepVQA feature, including features from reference and comrpessed videos
feat_video = [feat_compressed_mean(1:4) feat_reference_mean(1:4)...
    feat_compressed_mean(feat_temporal_ind) feat_reference_mean(feat_temporal_ind)];
% feature index 1-4: spatial features of compressed video
% 5-8: spatial features of reference video
% 9-12: temporal features of compressed video
% 13-16: temporal features of reference video

toc % compute time of feature extraction

%% SVR module
% The SVR model was trained on the whole LIVE Wild Compressed Video Quality Database
% using libsvm

load('svr_compressed_database.mat')

% the featuers needs to be rescaled since the training of SVR requires
% normalization of features
% rescaling parameters were computed on the whole LIVE Wild Compressed Video Quality Database
feat_video = a_para.*feat_video + b_para;
[predict_score, ~,~] = svmpredict(zeros(1), feat_video, model_reg)
