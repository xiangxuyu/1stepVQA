function feature_temporal = onestepVQA_temporal(frameNow, frameNext)

%% 1stepVQA Temporal Feature Computation %%

scalenum = 2;
window = fspecial('gaussian',7,7/6);
window = window/sum(sum(window));
C = 1;

shifts_GGD = [1 1; -1 1; 1 -1; -1 -1];
feat = [];

for itr_scale = 1 : scalenum
    
    for diff_iter = 1 : size(shifts_GGD, 1)
        
        %%%% displaced frame difference
        shift_now = shifts_GGD(diff_iter, :);
        frameNext_shifted = circshift(frameNext, shift_now);
        frameDiff_shifted = frameNow - frameNext_shifted;
        
        %%%% MSCN
        mu            = filter2(window, frameDiff_shifted, 'same');
        mu_sq         = mu.*mu;
        sigma         = sqrt(abs(filter2(window, frameDiff_shifted.*frameDiff_shifted, 'same') - mu_sq));
        structdis     = (frameDiff_shifted-mu)./(sigma+C);
        
        %%%% GGD
        [alpha overallstd]       = estimateggdparam(structdis(:));
        feat                     = [feat alpha overallstd^2];
    end
    
    %%%% downscale
    frameNow                 = imresize(frameNow, 0.5);
    frameNext                = imresize(frameNext, 0.5);
end

feature_temporal = feat;

end

