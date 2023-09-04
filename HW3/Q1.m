%%
% Load a random image from the web
url = 'C:\Users\LENOVO\Desktop\TV\TV.png';
img = imread(url);

% Convert the image to grayscale and double format
img = im2double(img);

% Add Gaussian noise with mean 0 and standard deviation 0.1
noise = 0.1 * randn(size(img));
img_noisy = img + noise;

% Set the parameters for the TV_Chambolle and TV_GPCL functions
lbd = 16; % fidelity parameter
alpha = 0.1; % fixed step length
NIT = 100; % maximum number of iterations
GapTol = 1e-4; % convergence tolerance for relative duality gap
verbose = true; % display iteration information or not

% Initialize the dual variables w1 and w2 as zeros
w1 = zeros(size(img));
w2 = zeros(size(img));

% Apply the TV_Chambolle function to denoise the image
[u_cham,w1_cham,w2_cham,Energy_cham,Dgap_cham,TimeCost_cham,itr_cham] = ...
      TV_Chambolle(w1,w2,img_noisy,lbd,alpha,NIT,GapTol,verbose);

% Apply the TV_GPCL function to denoise the image
[u_gpcl,w1_gpcl,w2_gpcl,Energy_gpcl,Dgap_gpcl,TimeCost_gpcl,itr_gpcl] = ...
      TV_GPCL(w1,w2,img_noisy,lbd,alpha,NIT,GapTol,verbose);

% Display the original, noisy and denoised images
figure;
subplot(2,2,1); imshow(img); title('Original image');
subplot(2,2,2); imshow(img_noisy); title(['Noisy image, PSNR = ', num2str(psnr(img_noisy,img)), ' dB']); 
% Calculate the PSNR of the denoised images by TV_Chambolle and TV_GPCL
psnr_cham = psnr(u_cham,img);
psnr_gpcl = psnr(u_gpcl,img);

% Display the PSNR values in the titles
subplot(2,2,3); imshow(u_cham); title(['Denoised by TV\_Chambolle, PSNR = ', num2str(psnr_cham), ' dB']);
subplot(2,2,4); imshow(u_gpcl); title(['Denoised by TV\_GPCL, PSNR = ', num2str(psnr_gpcl), ' dB']);


%%
%%%%%%%%%% function
function [u,w1,w2,Energy,Dgap,TimeCost,itr] = ...
      TV_Chambolle(w1,w2,f,lbd,alpha,NIT,GapTol,verbose)

n=length(f);                    % Assume a square image        
g=lbd*f;
gx = [g(:,2:n)-g(:,1:n-1), zeros(n,1)];
gy = [g(2:n,:)-g(1:n-1,:); zeros(1,n)];
sf = 0.5*lbd*sum(sum(f.^2));    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute Energy
DivW=([w1(:,1),w1(:,2:n)-w1(:,1:n-1)] + [w2(1,:);w2(2:n,:)-w2(1:n-1,:)]); 
Energy(1)=0.5*sum(sum((DivW-g).^2));

%Compute the primal u and the duality gap
u=f - (1/lbd)*DivW;   
ux = [u(:,2:n)-u(:,1:n-1), zeros(n,1)];
uy = [u(2:n,:)-u(1:n-1,:); zeros(1,n)];
gu_norm = sqrt(ux.^2+uy.^2);
Dgap(1)= sum(sum(gu_norm + ux.*w1 + uy.*w2));   
TimeCost(1) = 0;
t0 = cputime;                    %Start CPU clock

for itr=1:NIT
  % gradient of the objective function
  dFx = [DivW(:,1:n-1)-DivW(:,2:n), zeros(n,1)] + gx;
  dFy = [DivW(1:n-1,:)-DivW(2:n,:); zeros(1,n)] + gy;  
  % Chambolle's semi-implicit gradient descent method 
  w1 = w1- alpha * dFx;
  w2 = w2 - alpha * dFy;
  dFnorm = alpha * sqrt(dFx.^2+dFy.^2);
  w1 = w1 ./ (1.0 + dFnorm);
  w2 = w2 ./ (1.0 + dFnorm);
  
  % compute energy
  DivW=([w1(:,1),w1(:,2:n)-w1(:,1:n-1)] + [w2(1,:);w2(2:n,:)-w2(1:n-1,:)]); 
  Energy_new=0.5*sum(sum((DivW-g).^2));
  Energy(itr+1)=Energy_new;

  %Compute the primal u and the duality gap
  u  = f - (1/lbd)*DivW;   
  ux = [u(:,2:n)-u(:,1:n-1), zeros(n,1)];
  uy = [u(2:n,:)-u(1:n-1,:); zeros(1,n)];
  gu_norm = sqrt(ux.^2+uy.^2);
  Dgap(itr+1) = sum(sum(gu_norm + ux.*w1 + uy.*w2));  
  TimeCost(itr+1) = cputime-t0;
  
  % test for convergence:  
  % (Primal-Dual) / (|Primal|+|Dual|)< tol
  DualVal=sf-Energy_new/lbd; PriVal=DualVal+Dgap(itr+1);
  Dgap(itr+1) = Dgap(itr+1)/(abs(PriVal)+abs(DualVal));
  if verbose
    fprintf(1,' Chambolle itr %d: Obj %12.6e, rel dgap=%7.3e\n', ...
	itr, DualVal, Dgap(itr+1));
  end
  if (Dgap(itr+1) < GapTol )
    if verbose
      fprintf(1,'Chambolle: convergence tolerance reached: %6.2e\n',...
	  Dgap(itr+1));
    end
    break
  end
end
end
function [u,w1,w2,Energy,Dgap,TimeCost,itr] = ...
      TV_GPCL(w1,w2,f,lbd,alpha,NIT,GapTol,verbose)

n=length(f);                %Assume a square image        
g=lbd*f;
gx = [g(:,2:n)-g(:,1:n-1), zeros(n,1)];
gy = [g(2:n,:)-g(1:n-1,:); zeros(1,n)];
sf = 0.5*lbd*sum(sum(f.^2));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Compute energy
DivW=([w1(:,1),w1(:,2:n)-w1(:,1:n-1)] + [w2(1,:);w2(2:n,:)-w2(1:n-1,:)]); 
Energy(1)=0.5*sum(sum((DivW-g).^2));

%Compute the primal u and the duality gap
u=f - (1/lbd)*DivW;   
ux = [u(:,2:n)-u(:,1:n-1), zeros(n,1)];
uy = [u(2:n,:)-u(1:n-1,:); zeros(1,n)];
gu_norm = sqrt(ux.^2+uy.^2);
Dgap(1)= sum(sum(gu_norm + ux.*w1 + uy.*w2));   
TimeCost(1) = 0;
t0 = cputime;                %Start CPU clock

for itr=1:NIT
  % gradient of the objective function
  dFx = [DivW(:,1:n-1)-DivW(:,2:n), zeros(n,1)] + gx;
  dFy = [DivW(1:n-1,:)-DivW(2:n,:); zeros(1,n)] + gy;  

  % GP with constant step length
  w1 = w1 - alpha * dFx;
  w2 = w2 - alpha * dFy;
  wnorm= max(1, sqrt(w1.^2+w2.^2));
  w1 = w1./wnorm;
  w2 = w2./wnorm;
  
  % Compute Energy  
  DivW=([w1(:,1),w1(:,2:n)-w1(:,1:n-1)] + [w2(1,:);w2(2:n,:)-w2(1:n-1,:)]); 
  Energy_new=0.5*sum(sum((DivW-g).^2));
  Energy(itr+1)=Energy_new;
  %Compute the primal u and the duality gap
  u=f - (1/lbd)*DivW;   
  ux = [u(:,2:n)-u(:,1:n-1), zeros(n,1)];
  uy = [u(2:n,:)-u(1:n-1,:);zeros(1,n)];
  gu_norm = sqrt(ux.^2+uy.^2);
  Dgap(itr+1)= sum(sum(gu_norm + ux.*w1 + uy.*w2));    
  TimeCost(itr+1)=cputime-t0;

  % test for convergence:  
  % (Primal-Dual) / (|Primal|+|Dual|)< tol
  DualVal=sf-Energy_new/lbd; PriVal=DualVal+Dgap(itr+1);
  Dgap(itr+1) = Dgap(itr+1)/(abs(PriVal)+abs(DualVal));
  if verbose
    fprintf(1,' GPCL iter %4d: Obj=%11.6e, rel dgap=%7.3e\n', ...
	itr, DualVal, Dgap(itr+1));
  end
  if (Dgap(itr+1) < GapTol )
    if verbose
      fprintf(1,'GPCL convergence tolerance reached: %6.2e\n',...
	 Dgap(itr+1));
    end
    break
  end
  
end  
end

