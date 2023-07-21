function realign_all_niftis_unused(outputfolder, suffix)
%% main
if nargin < 1
    outputfolder = '/data2/MRI_PET_DATA/processed_images_final';
end
if nargin < 2
    suffix = '';
end

MRI_folder_init = ['/data2/MRI_PET_DATA/raw_data/MRI_nii' suffix filesep];
MRI_folder_new = [outputfolder '/ADNI_MRI_nii_recenter' suffix];

if ~exist(outputfolder,'dir')
    mkdir(outputfolder)
            
end
if ~exist(MRI_folder_new,'dir')
    mkdir(MRI_folder_new);
end

realign_nifti_local(MRI_folder_init, MRI_folder_new, 'mri');
end

function fnames = realign_nifti_local(raw_nifti_folder, folder_name, modality)
if ~exist(folder_name,'dir')
    mkdir(folder_name);
end

listing = dir(raw_nifti_folder);
disp(listing);
disp(raw_nifti_folder);
fnames = {};
for i=1:length(listing)
            
   if ~listing(i).isdir && strcmp(listing(i).name(end-3:end), '.nii')
        rid = regexp(listing(i).name,'^([0-9]{4}).*\.nii$','tokens');
        iid = regexp(listing(i).name,'^.*(I[0-9]+)_MRI\.nii$','tokens');
        fname = join([listing(i).folder filesep listing(i).name],'');
        system(['rsync -a ' fname ' ' folder_name '/../tmp/']);
        Vo = SetOriginToCenter([folder_name '/../tmp/' listing(i).name]);
        [data] = spm_read_vols(Vo);
        Vo.fname = [folder_name filesep rid{1}{1} '_' modality '_' iid{1}{1} '.nii'];
        fnames = [fnames, Vo.fname];
        if ~exist(Vo.fname, 'file')
            spm_write_vol(Vo, data);
        else
            disp(['skipping ' listing(i).folder filesep listing(i).name])
        end
   end
end

end