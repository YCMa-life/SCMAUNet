clear;
clc;
C=40;
N=4;
L=C/2;

allfilenames=dir(strcat('../data/ref_test/*.mat'));
filepathref='../data/ref_test/';    
filepathrecon='../SCMAU-Net/recon/C=40 N=4/';

d1=1;
n = length(allfilenames);
SSIM_all = zeros(n,1);
PSNR_all = zeros(n,1);
NMSE_all = zeros(n,1);


for num=1:n
    slicename=allfilenames(num).name;   

    fileslice=slicename(1:end-4);
    filenameref=strcat(filepathref,slicename);   
    load (filenameref);                                
    imagsref=imags; 

    dim=size(imags);
    Yres=dim(1);
    Xres=dim(2);
    C=dim(3);

    filenamerecon=strcat(filepathrecon,fileslice,'.mat');
    load (filenamerecon);
    imagsrecon=imags;

    for ch=1:C/2
        Image=imagsref(:,:,ch)+i*imagsref(:,:,ch+C/2); 

        S0=fftshift(fft2(ifftshift(Image)))/sqrt(Xres*Yres);
        E0=imagsrecon(:,:,ch)+i*imagsrecon(:,:,ch+C/2); 
        
        Finv=zeros(Yres,Xres);
        S0Rec=fftshift(fft2(ifftshift(E0)))/sqrt(Xres*Yres);            
        for m=1:(Yres/2-L)     
            Finv(m,:)=S0Rec(m,:)./(exp(i*2*pi*(m-1+Yres/2)/Yres)-1);
        end 

        for m=(Yres/2+L+1):Yres
            Finv(m,:)=S0Rec(m,:)./(exp(i*2*pi*(m-1-Yres/2)/Yres)-1);
        end 
        %%% Replacing with acturally acquired data %%%
        for m=1:N:(Yres-d1)
            Finv(m+d1,:)=S0(m+d1,:);
        end
        %%% Replacing with center true data %%%
        Finv(Yres/2-L+1:Yres/2+L,:)=S0(Yres/2-L+1:Yres/2+L,:); 

        % Step10: Reconstracted Image
        Rec=sqrt(Yres*Xres)*ifftshift(ifft2(fftshift(Finv)));   

        imagsrecon(:,:,ch)=real(Rec);
        imagsrecon(:,:,ch+16)=imag(Rec); 
    end

    
    Image=zeros(Yres,Xres);
    Rec=zeros(Yres,Xres);  

    for ch=1:16

        Image3D(:,:,ch)=(imagsref(:,:,ch)+i*imagsref(:,:,ch+16)); 
        Rec3D(:,:,ch)=(imagsrecon(:,:,ch)+i*imagsrecon(:,:,ch+16)); 

    end
    
    Image=VirtualReferenceCoil(Image3D);  
    Rec=VirtualReferenceCoil(Rec3D);  

    Image=Image/max(abs(Image(:)));
    Rec=Rec/max(abs(Rec(:))); 

    BW=ProduceBrainMask(slicename); 

    ImageMask=Image.*BW;
    RecMask=Rec.*BW;

    VSSIM=ssim(double(abs(RecMask)),double(abs(ImageMask)));
    VPSNR=psnr(RecMask,ImageMask);
    VNMSE=immse(RecMask,ImageMask);

    fprintf('name=%s\n',slicename);
    
    fprintf('NMSE = %.6f\n', VNMSE);

    fprintf('PSNR = %.6f\n', VPSNR);

    fprintf('SSIM = %.6f\n', VSSIM);

    SSIM_all(num) = VSSIM;
    PSNR_all(num) = VPSNR;
    NMSE_all(num) = VNMSE;
 
end

SSIM_mean = mean(SSIM_all);
PSNR_mean = mean(PSNR_all);
NMSE_mean = mean(NMSE_all);
SSIM_std = std(SSIM_all);
PSNR_std = std(PSNR_all);
NMSE_std = std(NMSE_all);

fprintf('SSIM mean = %.6f\n', SSIM_mean);
fprintf('PSNR mean = %.6f\n', PSNR_mean);
fprintf('NMSE mean = %.6f\n', NMSE_mean);
fprintf('SSIM std = %.6f\n', SSIM_std);
fprintf('PSNR std = %.6f\n', PSNR_std);
fprintf('NMSE std = %.6f\n', NMSE_std);



