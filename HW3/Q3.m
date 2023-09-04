% Read the image
im = imread('C:\Users\LENOVO\Desktop\image3.png');

% Convert to grayscale if needed
if ndims(im) == 3
    im = rgb2gray(im);
end
% Convert to double type
im = im2double(im);

% Add Gaussian noise
noise = 0.05 * randn(size(im)); % adjust the noise level here
im_noisy = im + noise;

% Denoise using isotropic diffusion
lambda = 0.1; % adjust the diffusion speed here
constant = 1.8; % adjust the conduction coefficient here
im_isodiff = isodiff(im_noisy, lambda, constant);

% Denoise using anisotropic diffusion
niter = 2; % adjust the number of iterations here
kappa = 25; % adjust the conduction coefficient here
lambda = 0.18; % adjust the diffusion speed here
option = 2; % choose between diffusion equation 1 or 2
im_anisodiff = anisodiff(im_noisy, niter, kappa, lambda, option);

% Display the results
figure;
subplot(2,2,1); imshow(im); title('Original image');
subplot(2,2,2); imshow(im_noisy); title('Noisy image');
subplot(2,2,3); imshow(im_isodiff); title('Isotropic diffusion');
subplot(2,2,4); imshow(im_anisodiff); title('Anisotropic diffusion');


% Evaluate NIQE score for each image using the default model
niqe_im = niqe(im); % lower score indicates better perceptual quality
niqe_im_noisy = niqe(im_noisy);
niqe_im_isodiff = niqe(im_isodiff);
niqe_im_anisodiff = niqe(im_anisodiff);

% Evaluate SSIM index for each image using the original image as reference
ssim_im_noisy = ssim(im_noisy, im); % higher index indicates better similarity
ssim_im_isodiff = ssim(im_isodiff, im);
ssim_im_anisodiff = ssim(im_anisodiff, im);

% Display the results in a table
T = table ([niqe_im; niqe_im_noisy; niqe_im_isodiff; niqe_im_anisodiff], ...
           [NaN; ssim_im_noisy; ssim_im_isodiff; ssim_im_anisodiff], ...
           'RowNames' , {'Original' , 'Noisy' , 'Isotropic' , 'Anisotropic'}, ...
           'VariableNames' , {'NIQE' , 'SSIM'});
disp(T)
%%
function diff = anisodiff(im, niter, kappa, lambda, option)

if ndims(im)==3
  error('Anisodiff only operates on 2D grey-scale images');
end

im = double(im);
[rows,cols] = size(im);
diff = im;
  
for i = 1:niter
%   fprintf('\rIteration %d',i);

  % Construct diffl which is the same as diff but
  % has an extra padding of zeros around it.
  diffl = zeros(rows+2, cols+2);
  diffl(2:rows+1, 2:cols+1) = diff;

  % North, South, East and West differences
  deltaN = diffl(1:rows,2:cols+1)   - diff;
  deltaS = diffl(3:rows+2,2:cols+1) - diff;
  deltaE = diffl(2:rows+1,3:cols+2) - diff;
  deltaW = diffl(2:rows+1,1:cols)   - diff;

  % Conduction
  if option == 1
    cN = exp(-(deltaN/kappa).^2);
    cS = exp(-(deltaS/kappa).^2);
    cE = exp(-(deltaE/kappa).^2);
    cW = exp(-(deltaW/kappa).^2);
  elseif option == 2
    cN = 1./(1 + (deltaN/kappa).^2);
    cS = 1./(1 + (deltaS/kappa).^2);
    cE = 1./(1 + (deltaE/kappa).^2);
    cW = 1./(1 + (deltaW/kappa).^2);
  end

  diff = diff + lambda*(cN.*deltaN + cS.*deltaS + cE.*deltaE + cW.*deltaW);

%  Uncomment the following to see a progression of images
%  subplot(ceil(sqrt(niterations)),ceil(sqrt(niterations)), i)
%  imagesc(diff), colormap(gray), axis image

end
end
function diff = isodiff(im,lambda,constant)

if ndims(im)==3
  error('Isodiff only operates on 2D grey-scale images');
end

im = double(im);
[rows,cols] = size(im);
diff = im;

% Construct diffl which is the same as diff but
% has an extra padding of zeros around it.
diffl = zeros(rows+2, cols+2);
diffl(2:rows+1, 2:cols+1) = diff;

% North, South, East and West differences
deltaN = diffl(1:rows,2:cols+1)   - diff;
deltaS = diffl(3:rows+2,2:cols+1) - diff;
deltaE = diffl(2:rows+1,3:cols+2) - diff;
deltaW = diffl(2:rows+1,1:cols)   - diff;

% Conduction
cN = constant;
cS = constant;
cE = constant;
cW = constant;

diff = diff + lambda*(cN.*deltaN + cS.*deltaS + cE.*deltaE + cW.*deltaW);

%  Uncomment the following to see a progression of images
%  subplot(ceil(sqrt(niterations)),ceil(sqrt(niterations)), i)
%  imagesc(diff), colormap(gray), axis image
end
