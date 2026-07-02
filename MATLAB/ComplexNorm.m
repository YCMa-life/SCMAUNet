function imags=ComplexNorm(imags)

 phase=angle(imags);
 mag = abs(imags);
 
 mag_max=max(mag(:));
 mag_min=min(mag(:));
 mag=(mag - mag_min)/(mag_max-mag_min);

 imags=mag.*exp(i*phase);
%  figure;imshow(abs(imags),[]);
%  figure;imshow(angle(imags),[]);