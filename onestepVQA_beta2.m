clc
clear
close all

addpath('Function')

tic

frameheight = 1080;
framewidth = 1920;


%%%% reference video and test video
name_reference = 'A001.yuv';
name_compressed = 'A001-360p-37-1080p.yuv';

filep            = dir(name_reference);
fileBytes        = filep.bytes;
fwidth           = 0.5;
fheight          = 0.5;
clear filep
Nframes      = fileBytes/(framewidth*frameheight*(1+2*fheight*fwidth));

for frame_ind = 1 : 1 : (Nframes - 1)
    
    fprintf('reference video name: %s;\n Current frame: %i\n', name_reference, frame_ind);
    
    %%%% read current and next frames
    frameNow             = read_single_frame(name_reference, frame_ind, frameheight, framewidth);
    frameNext            = read_single_frame(name_reference, (frame_ind + 1), frameheight, framewidth);
    
    %%%% compute spatial and temporal feature
    feat_spatial_now     = onestepVQA_spatial_7_29(frameNow);
    feat_temporal_now    = onestepVQA_temporal_7_29(frameNow, frameNext);
    
    feat_reference(frame_ind, :) = [feat_spatial_now feat_temporal_now];
end

feat_reference_mean = mean(feat_reference);




filep            = dir(name_compressed);
fileBytes        = filep.bytes;
fwidth           = 0.5;
fheight          = 0.5;
clear filep
Nframes      = fileBytes/(framewidth*frameheight*(1+2*fheight*fwidth));


for frame_ind = 1 : 1 : (Nframes - 1)
    
    fprintf('Compressed video name: %s;\n Current frame: %i\n', name_compressed, frame_ind);
    
    %%%% read current and next frames
    frameNow             = read_single_frame(name_compressed, frame_ind, frameheight, framewidth);
    frameNext            = read_single_frame(name_compressed, (frame_ind + 1), frameheight, framewidth);
    
    %%%% compute spatial and temporal feature
    feat_spatial_now     = onestepVQA_spatial_7_29(frameNow);
    feat_temporal_now    = onestepVQA_temporal_7_29(frameNow, frameNext);
    
    feat_compressed(frame_ind, :) = [feat_spatial_now feat_temporal_now];
end

feat_compressed_mean = mean(feat_compressed);

alpha_reference = [feat_reference_mean(5) feat_reference_mean(7) feat_reference_mean(9) feat_reference_mean(11)];
d_max = find(feat_reference_mean == max(alpha_reference));

feat_video_ind = [1:4 d_max d_max+1 d_max+8 d_max+9];


%%%% 1stepVQA feature
feat_video = [feat_compressed_mean(feat_video_ind) feat_reference_mean(feat_video_ind)];


toc