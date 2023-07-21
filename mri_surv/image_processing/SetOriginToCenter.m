    function [Vo, origin]=SetOriginToCenter(P)
    V=spm_vol(P);
    cur_dim =V.dim;
    value=[cur_dim./2];
    mat=V(1).mat;
    mat(1:3,4)=-mat(1:3,1:3)*value(1:3)'; %The last column of this matrix sets the origin. It is calculated by multiplying
                                                       % the voxel sizes (stored in mat(1,1), mat(2,2), mat(3,3)) by the voxel numbers
                                                       % to get the milimetric position of the origin in the volume (this is my take on
                                                       % a more sophisticated explanation by John Ashburner)
    M  = mat;
    if spm_flip_analyze_images, M = diag([-1 1 1 1])*M; end;
    vx = sqrt(sum(M(1:3,1:3).^2));
    if det(M(1:3,1:3))<0, vx(1) = -vx(1); end;
    origin = M\[0 0 0 1]';
    origin = round(origin(1:3));
    [V.mat] = deal(mat);
    switch spm('ver')
        case {'SPM12b' 'SPM12'}
            Vo = spm_create_vol(V);
        case {'SPM5','SPM8', 'SPM8b'}
            Vo = spm_create_vol(V, 'noopen');
        otherwise
    	error(sprintf('SPM v%s is not supported!', spm('ver')));

    end
