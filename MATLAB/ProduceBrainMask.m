function BW=ProduceBrainMask(slicename)

    fileslice=slicename(1:end-4);

    filepathref='../data/ref_test/';
    filenameref=strcat(filepathref,slicename);   
    load (filenameref);
    
    imagecom=zeros(320,320);
    for ch=1:16
          imref=imags(:,:,ch)+j*imags(:,:,ch);
          imagecom=imagecom+abs(imref).^2;  
    end
    
    imagecom=abs(sqrt(imagecom));
    T = graythresh(imagecom);
    BW=single(imagecom>T);
    
    BW= imfill(BW,'holes');








