function ProducePMRIdata()

allfilenames=dir(strcat(pwd,'\MultiCoilsData\*.mat'));
for num=1:16  %%% total number of slices, set 16 for demonstration.
    filename=allfilenames(num).name;
    filenamenew=strcat(pwd,'\MultiCoilsData\',filename);  
    load(filenamenew);
    
    dim=size(Image3D);
    Yres=dim(1);
    Xres=dim(2);
    Cres=dim(3); %% Coil numbers
    
    N=4;   %%% N: skip size, for example, can be changed in range 2-10 for different accerlation rate.
    C=40;  %%% C: central full sampled k-space lines.
    L=C/2;
    
    Mask=zeros(Yres,Xres);
    Mask(1:N:end,:)=1;
    Mask(Yres/2-L+1:Yres/2+L,:)=1;
%   Xres*Yres/sum(Mask(:))   
     
    for ch=1:Cres; 
        data1=Image3D(:,:,ch);  
        temp1=ComplexNorm(data1);

        %%% Reconstructed with full sampled data, obtain referenced complex images for CUNET and FDCUNET 
        imags1(:,:,ch)=real(temp1);
        imags1(:,:,16+ch)=imag(temp1);
        
        %%%  Referenced complex images for SCUNET
        temp2=temp1([2:end,1],:)-temp1(1:end,:);    %% Complex difference transform       
        imags2(:,:,ch)=real(temp2);
        imags2(:,:,16+ch)=imag(temp2);

        kdata=fftshift(fft2(data1));
        data=kdata.*Mask;
        image=ifft2(ifftshift(data)); 
        temp3=ComplexNorm(image); 
       
        %%% Undersampled complex images for CUNET and FDCUNET 
        imags3(:,:,ch)=real(temp3);
        imags3(:,:,16+ch)=imag(temp3);   
        
        %%% Undersampled complex images for SCUNET 
        temp4=temp3([2:end,1],:)-temp3(1:end,:);     %% Complex difference transform  
        imags4(:,:,ch)=real(temp4);
        imags4(:,:,16+ch)=imag(temp4);
    end
    
    str=pwd;
    index_dir=findstr(str,'\');
    strtemp=str(1:index_dir(end)-1);
    
%     imags=single(imags1);
%     filename1=strcat(strtemp,'\SCUNET\data\ref_PMRI_CUNET\',filename); 
%     save(filename1,'imags');  

    imags=single(imags2);
    filename2=strcat(strtemp,'\SCUNET\data\ref_PMRI_SCUNET\',filename); 
    save(filename2,'imags'); 

%     imags=single(imags3);               
%     filename3=strcat(strtemp,'\SCUNET\data\train_PMRI_CUNET\',filename);  %% Undersampled complex images for CUNET and FDCUNET
%     save(filename3,'imags');

    imags=single(imags4);               
    filename4=strcat(strtemp,'\SCUNET\data\train_PMRI_SCUNET\',filename); %% Undersampled complex images for SCUNET
    save(filename4,'imags');
end








