
function readmulticoildata()
allfilenames=dir(strcat(pwd,'\RawData\*.h5'));
for num=1:1   %%% Total number of downloaded data
    filename=allfilenames(num).name;
    % h5disp(filename,'/');
    filenamenew=strcat(pwd,'\RawData\',filename)
    
    % h5disp(filenamenew,'/')
    kspace = h5read(filenamenew,'/kspace'); 
    ismrmrd_header =h5read(filenamenew,'/ismrmrd_header'); 

%     reconstruction_esc =h5read(filename,'/reconstruction_esc'); 
%     reconstruction_rss =h5read(filenamenew,'/reconstruction_rss'); 
%     rec=reconstruction_rss(:,:,1);   
%     figure;imshow(abs(rec),[]) 
 
    kspace_r = getfield(kspace,'r');               % real part
    kspace_i = getfield(kspace,'i');               % imaginary part

    multicoildata= complex(kspace_r,kspace_i);      % complex data
        
    dim=size(multicoildata);
    Yres=dim(1); 
    Xres=dim(2);
    Cres=dim(3);    %% Coils
    Zres=dim(4);    %% Slices
    if((Yres==320)&(Xres==640)&(Cres==16)&(Zres==16))
        for z=1:Zres
            data3D=multicoildata(:,:,:,z);
            Image3D=ifftshift(ifft2(fftshift(data3D)));
            Image3D(:,end-159:end,:)=[]; 
            Image3D(:,1:160,:)=[]; 
            Image3D=single(Image3D);
            
%             for j=1:16
%               figure(j);imshow(abs(Image3D(:,:,j)),[]);
%               figure(j+16);imshow(angle(Image3D(:,:,j)),[]);
%             end  

            recname=strcat(pwd,'\MultiCoilsData\brainmulticoil_file',num2str(num),'_slice',num2str(z),'.mat');
            save (recname,'Image3D');
        end
    else
        disp('skip');
    end
end