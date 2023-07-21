function rids = realign_all_niftis_nacc(outputfolder, suffix)
%% main
if nargin < 1
    outputfolder = '/data2/MRI_PET_DATA/processed_images_final';
end
if nargin < 2
    suffix = '';
end

MRI_folder_init = '/data2/MRI_PET_DATA/raw_data/NACC_MRI';
fnames = dir(MRI_folder_init);
fnames = {fnames.name};
rids = arrayfun(@(x) regexp(x,'^(NACC[0-9]+)\.nii$','tokens'), fnames, 'uniformoutput', false);
rids = [rids{:}];
rids = [rids{:}];
rids_mri = [rids{:}];
disp(rids_mri)
rids = rids_mri;

MRI_folder_new = [outputfolder '/ADNI_MRI_NACC_recenter' suffix];

if ~exist(outputfolder,'dir')
    mkdir(outputfolder)
end
if ~exist(MRI_folder_new,'dir')
    mkdir(MRI_folder_new);
end

realign_nifti_nacc(MRI_folder_init, MRI_folder_new, rids, 'mri');

function fnames = realign_nifti_nacc(raw_nifti_folder, folder_name, rid_list, modality)
if ~exist(folder_name,'dir')
    mkdir(folder_name);
end

listing = dir(raw_nifti_folder);
fnames = {};
for i=1:length(listing)
   if ~listing(i).isdir && strcmp(listing(i).name(end-3:end), '.nii')
        rid = regexp(listing(i).name,'^(NACC[0-9]+)\.nii$','tokens');
        fname = join([listing(i).folder filesep listing(i).name],'');
        system(['rsync -a ' fname ' ' folder_name '/../tmp/']);
        if ismember(rid{1},rid_list)
            Vo = SetOriginToCenter([folder_name '/../tmp/' listing(i).name]);
            [data] = spm_read_vols(Vo);
            Vo.fname = [folder_name filesep rid{1}{1} '_' modality '.nii'];
            fnames = [fnames, Vo.fname];
            if ~exist(Vo.fname, 'file')
                spm_write_vol(Vo, data);
            end
        else
            disp(['file ' listing(i).name ' not in list of RIDs'])
        end
   else
       disp(['skipping ' listing(i).folder filesep listing(i).name])
end

end