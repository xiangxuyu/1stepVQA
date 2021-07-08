function feature_spatial = onestepVQA_spatial(frameNow)

%% 1stepVQA Spatial Feature Computation %%

scalenum = 2;
window = fspecial('gaussian',7,7/6);
window = window/sum(sum(window));
C = 1;

feat = [];

for itr_scale = 1 : scalenum
    %%%% MSCN
    mu            = filter2(window, frameNow, 'same');
    mu_sq         = mu.*mu;
    sigma         = sqrt(abs(filter2(window, frameNow.*frameNow, 'same') - mu_sq));
    structdis     = (frameNow-mu)./(sigma+C);
    
    %%%% GGD
    [alpha overallstd]       = estimateggdparam(structdis(:));
    feat                     = [feat alpha overallstd^2];
    
    %%%% downscale
    frameNow                   = imresize(frameNow,0.5);
end

feature_spatial = feat;

end

