%%% Loading Data
clear all
load('data.mat')
%%
%%% Reading images and save them as jpg
% Image must be single precision.
imageData = single(imageData);
% Display it.
imshow(imageData, 'InitialMagnification', 1000)
title('Image Data')
imwrite(imageData,'imageData.jpg','jpg');
imageMask = single(imageMask);
% Display it.
figure
imshow(imageMask, 'InitialMagnification', 1000)
title('Image Mask')
imwrite(imageMask,'imageMask.jpg','jpg');
%%
%%% 2.2

k=3;
q=5;
b_const = 1;
b_init = b_const .* imageMask;
f = 9;
sigma_kernal = 2.5;
eps_kmeans = 1e-7;
eps_iterate = 1e-5;
N_max = 200;
%%
% Define the size of the Gaussian kernel
kernel_size = [f, f];

% Define the standard deviation for Gaussian Kernel
sigma = sigma_kernal;

% Create Gaussian kernel
G = fspecial('gaussian', kernel_size, sigma);
%%
%%% 2.3
[seg, mu, sigma] = KMeans(imageData, imageMask, k, eps_kmeans);
showSegmented(seg, k, "Segmentation after kmeans", "C:\Users\LENOVO\Desktop\seg_kmeans")
%%
%%% 2.4.1

J_init = 10000000000000;
[u, b, c, J] = iterate(imageData, imageMask, seg, b_init, mu, q, G, J_init, eps_iterate, N_max);
%%
%%% 2.4.2

t = 1:1:200;
plot(t,J')
title('Loss Function In Different Step')
xlabel('n')
ylabel('Loss')
%%
%%% 2.4.4
showSegmented(u, k, "Final segmentation", "C:\Users\LENOVO\Desktop\final_seg")
%%
%%% 2.4.3
imshow(b, 'InitialMagnification', 1000)
title('Bias Field')
%%
%%% 2.4.4
A = computeA(u, c);
imshow(A, 'InitialMagnification', 1000)
title('Bias Removed Image')
%%
%%% 
imshow(imageData-b_init, 'InitialMagnification', 1000)
title('Bias Removed Image')
%%
function [seg, mu, sigma] = KMeans(img, mask, k, eps)

    % Set seed to ensure reproducibility
    rng(1);

    % Initialize
    [R, C] = size(img);
    sigma  = zeros(k, 1); 
    
    % Initialize mu by randomly selecting k points from the foreground (To-do: Better initialization scheme)
    fg = img(mask == 1);
    mu = datasample(fg, k, 'Replace', false);
    
    % Compute the initial segmentation (todo: vectorize)
    seg = zeros(size(img));
    for r = 1:R
        for c = 1:C
            if (mask(r, c) == 1)
                [~, idx]  = min(abs(mu - img(r, c)));
                seg(r, c) = idx;
            end
        end
    end
    
    % Iteratively update mus
    n_iter = 0;
    flag = true;
    while flag
        prev_loss = 0;
        new_loss = 0;
        % Update mu and compute energies
        for i = 1:k
            mask_i = (seg == i);
            class_i = img(mask_i);

            % Compute loss before update
            diff = (class_i - mu(i)).^2;
            prev_loss = prev_loss + sum(diff(:));

            % Update the ith mean and compute the corresponding sigma
            mu(i) = mean(class_i(:));
            sigma(i) = std(class_i(:));

            % Update labels
            for r = 1:R
                for c = 1:C
                    if (mask(r, c) == 1)
                        [~, idx] = min(abs(mu - img(r, c)));
                        seg(r, c) = idx;
                    end
                end
            end

            % Compute the new mask
            mask_i  = (seg == i);
            class_i = img(mask_i);

            % Compute loss after update
            diff = (class_i - mu(i)).^2;
            new_loss = new_loss + sum(diff(:));
        end
        
        % Stopping condition
        rel_change = abs(new_loss - prev_loss) / prev_loss;
        if rel_change <= eps
            flag = false;
        end
        
        n_iter = n_iter + 1;
        
    end
    
    fprintf("K-Means ran for %d iterations\n", n_iter);

end
function J = objectiveFunction(img, b, c, q, u, w)
    
    K = size(c, 1);
    
    temp = u.^q;
    sum_term_1 = sum(temp, 3);
    temp = (img.^2) .* sum_term_1;
    term_1  = conv2(temp, w, "same");
    
    sum_term_2 = zeros(size(img)); % \sum_{k=1}^K u_{jk}^q * c_k^2
    sum_term_3 = zeros(size(img)); % \sum_{k=1}^K u_{jk}^q * c_k
    
    for k = 1:K
        temp = (u(:,:,k).^q) .* c(k);
        sum_term_3 = sum_term_3 + temp;
        sum_term_2 = sum_term_2 + (temp .* c(k));
    end
    
    % Convolve with w
    term_2  = (b.^2) .* conv2(sum_term_2, w, "same"); 
    temp = img .* sum_term_3; % y_j * \sum_{k=1}^K u_{jk}^q * c_k
    term_3  = (2*b) .* conv2(temp, w, "same");
    
    temp = term_1 + term_2 - term_3;
    J = sum(temp(:));
    
end
function b = updateB(img, mask, c, u, q, w)
    
    K = size(c, 1);

    sum_num = zeros(size(img)); % \sum_{k=1}^K u_{jk}^q * c_k
    sum_den = zeros(size(img)); % \sum_{k=1}^K u_{jk}^q * c_k^2

    for k = 1:K
        temp = (u(:,:,k).^q) .* c(k);
        sum_num = sum_num + temp;
        sum_den = sum_den + (temp .* c(k));
    end
    
    % Convolve with w
    temp = img .* sum_num; % y_j * \sum_{k=1}^K u_{jk}^q * c_k
    num  = conv2(temp, w, "same");
    den  = conv2(sum_den, w, "same"); % 
    
    b = num ./ den;
    b(isnan(b)) = 0; % Set NaN to 0 that arise due to 0/0 in the background
    b = b .* mask;
    
end
function c = updateC(img, mask, b, u, q, w)
    
    sum_num = conv2(b, w, "same"); % \sum_{i=1}^N w_{ij}*b_i
    sum_den = conv2(b.^2, w, "same"); % \sum_{i=1}^N w_{ij}*b_i^2
    
    temp = u .^ q; % [256, 256, 3]
    num  = temp .* (img .* sum_num .* mask); % [256, 256, 3]
    num  = squeeze(sum(sum(num, 1), 2)); % Sum over first 2 dimensions, output is of shape [1, 1, 3], squeeze it
    
    den  = temp .* (sum_den .* mask); % [256, 256, 3]
    den  = squeeze(sum(sum(den, 1), 2)); % Sum over first 2 dimensions, output is of shape [1, 1, 3], squeeze it
    
    c = num ./ den;
    
end
function u = updateU(d, q, mask)

    num = (1 ./ d) .^ (q - 1); % [256, 256, 3]
    den = sum(num, 3); % [256, 256, 3]
    u = num ./ den; % [256, 256, 3] Uses broadcasting
    u(isnan(u)) = 0; % Set NaN to 0 that arise due to 0/0 in the background
    u = u .* mask; % [256, 256, 3] Uses broadcasting
    
end
function [u, b, c, J] = iterate(img, mask, u, b, c, q, w, J_init, eps, N_max)
    
    i_fcm = 0;
    J_after = J_init;
    J = zeros(N_max, 1);
    % J(1) = J_init; % The initial J is very large (due to poor segementation by K-Means) thus it skews the graph for J vs iters a lot. Thus we don't plot this value in the graph 
    while true
        
        d = distance(img, mask, b, c, w);
        u = updateU(d, q, mask);
        b = updateB(img, mask, c, u, q, w);
        c = updateC(img, mask, b, u, q, w);
        J_before = J_after;
        J_after = objectiveFunction(img, b, c, q, u, w);
        J(i_fcm + 1) = J_after;
        
        fprintf("Before: %f | After: %f\n", round(J_before, log10(1/eps)), round(J_after, log10(1/eps)));
        if  (abs(J_before - J_after)/J_before) <= eps 
            fprintf("Stopping\n");
            break;
        end
        
        i_fcm = i_fcm + 1;
        fprintf("Iteration %d completed\n\n", i_fcm);
        if i_fcm >= N_max
            break
        end
        
    end
    
end
function showSegmented(seg, k, title_str, path)
    % Custom color map
    map = [0 0 0; 1 0 0; 0 1 0; 0 0 1];
    % Normalize the image to be in [0, 1]
    seg = seg ./ k; 
    % Show and save the image
    fig = imshow(seg);
    colormap(map);
    title(title_str);
    saveas(fig, path, "jpg");
end
function A = computeA(u, c)
    [R, C, K] = size(u);
    A = zeros(R, C);
    for k = 1:K
        A = A + u(:, :, k) .* c(k);
    end 
end
function d = distance(img, mask, b, c, w)
    
    [R, C] = size(img);
    K = size(c, 1);
    
    d = zeros(R, C, K);
    temp_1 = conv2(b, w, "same");
    temp_2 = conv2(b.^2, w, "same");
    for k = 1:K
        term_1 = (img.^2) .* sum(w(:));
        term_2 = (-2) .* img .* c(k) .* temp_1;
        term_3 = (c(k)^2) .* temp_2;
        d(:, :, k) = term_1 + term_2 + term_3;
    end
    d = d .* mask;
    
end
