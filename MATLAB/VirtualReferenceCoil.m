%%% Implement of Virtual Reference Coil algorithm (Parker DL, etc.Magn Reson Med 2014;72:536-569.)
%%% Obtain single complex image using multicoil complex images
%%% Jin Z, Xiang QS. Improving accelerated MRI by deep learning with sparsified complex data. Magn Reson Med. doi:10.1002/mrm.29556
%%% Zhaoyang Jin, 2022-11-22

function Rec=VirtualReferenceCoil(Image3D)
dim=size(Image3D);
Yres=dim(1);
Xres=dim(2);
Cres=dim(3);

VC=zeros(Yres,Xres);
for ch=1:Cres  

    x0=Image3D(108:148,140:180,ch);
    test=mean(x0(:));  
    ThetaRef(ch)=conj(test/abs(test));  

end

for ch=1:Cres  
    for m=1:Yres
        for n=1:Xres
            sigma=abs(Image3D(m,n,:));
            WM=abs(Image3D(m,n,ch))/sum(sigma(:)); 
            VC(m,n)=VC(m,n)+WM*Image3D(m,n,ch).*ThetaRef(ch);
        end
    end  
end

VC=VC./abs(VC);
test1=zeros(Yres,Xres);
test2=zeros(Yres,Xres);
LP=zeros(Yres,Xres);
LP(Yres/2-69:Yres/2+70,Xres/2-69:Xres/2+70)=1;
for ch=1:Cres   
    test1=VC.*conj(Image3D(:,:,ch));   
    test2=ifft2(fftshift(fft2(test1)).*LP);    
    test2=test2./abs(test2);   
    Image3D(:,:,ch)=Image3D(:,:,ch).*test2; 
end

Rn=zeros(Cres,Cres);
for m=1:Cres
    for n=1:Cres
        temp1=Image3D(1:20,1:20,m);
        temp2=conj(Image3D(1:20,1:20,n));
        temp3=temp1(:).*temp2(:);
        Rn(m,n)=mean(temp3(:)); 
    end
end
Rn=inv(Rn); 
Rn=abs(Rn);   

Rec=zeros(Yres,Xres);
temp=zeros(Cres,1);
for m=1:Yres
    for n=1:Xres
        temp(:)=Image3D(m,n,:);
        temp=Rn*temp;
        Rec(m,n)=mean(temp(:));
    end
end  
  
for m=1:2:Yres
    Rec(m,:)=-Rec(m,:);
end
for n=1:2:Xres
    Rec(:,n)=-Rec(:,n);
end