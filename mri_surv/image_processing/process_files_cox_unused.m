function process_files_cox_unused(suffix, reg_exp)
addpath(genpath('/home/mfromano/spm/spm12/'));
BASE_DIR = ['/data2/MRI_PET_DATA/processed_images_final' suffix filesep];
MRI_folder_old = [BASE_DIR 'ADNI_MRI_nii_recenter' suffix filesep];

disp(MRI_folder_old)
if nargin < 2
    reg_exp = '^([0-9]+_mri_I[0-9]+).*\.nii$';
end

fnames = dir(MRI_folder_old);
fnames = {fnames.name};
disp(fnames)
rids = arrayfun(@(x) regexp(x,reg_exp,'tokens'), fnames, 'uniformoutput', false);
rids = [rids{:}];
rids = [rids{:}];
rids = [rids{:}];
brain_stripped_dir = ['/data2/MRI_PET_DATA/processed_images_final' suffix filesep 'brain_stripped' suffix];
new_mri_dir = ['/data2/MRI_PET_DATA/processed_images_final' suffix filesep 'mri_processed' suffix];
mkdir(new_mri_dir)
mkdir(brain_stripped_dir);
rand('state',10);
maxNumCompThreads = 1;
spm('defaults', 'PET');
spm_jobman('initcfg');
parpool(10);
cat12('expert')
parfor i=1:length(rids)
    if exist(['/data2/MRI_PET_DATA/processed_images_final' suffix '/brain_stripped' suffix '/masked_brain_mri_' rids{i} '.nii'], 'file')
        disp(['Skipping ' rids{i} ])
        continue
    end
    rand('state',10);
    disp(['rsync -v ' MRI_folder_old rids{i} '.nii ' new_mri_dir filesep])
    system(['rsync -v ' MRI_folder_old rids{i} '.nii ' new_mri_dir filesep]);
    try
        jobs = batch_process_mri(rids{i}, suffix, new_mri_dir);
        spm_jobman('run_nogui', jobs);
    catch
        disp(['could not process ' rids{i} ]);
    end

end

end
%-----------------------------------------------------------------------
% Job saved on 22-Oct-2020 12:09:13 by cfg_util (rev $Rev: 7345 $)
% spm SPM - SPM12 (7771)
% cfg_basicio BasicIO - Unknown
%-----------------------------------------------------------------------
function matlabbatch = batch_process_mri(rid, suffix, old_dir)
%-----------------------------------------------------------------------
% Job saved on 21-Apr-2021 20:19:18 by cfg_util (rev $Rev: 7345 $)
% spm SPM - SPM12 (7771)
% cfg_basicio BasicIO - Unknown
%-----------------------------------------------------------------------
matlabbatch{1}.spm.spatial.preproc.channel.vols = {[old_dir '/' rid '.nii,1']};
matlabbatch{1}.spm.spatial.preproc.channel.biasreg = 0.001;
matlabbatch{1}.spm.spatial.preproc.channel.biasfwhm = 60;
matlabbatch{1}.spm.spatial.preproc.channel.write = [0 1];
matlabbatch{1}.spm.spatial.preproc.tissue(1).tpm = {'/usr/local/spm/spm12/tpm/TPM.nii,1'};
matlabbatch{1}.spm.spatial.preproc.tissue(1).ngaus = 1;
matlabbatch{1}.spm.spatial.preproc.tissue(1).native = [0 0];
matlabbatch{1}.spm.spatial.preproc.tissue(1).warped = [1 0];
matlabbatch{1}.spm.spatial.preproc.tissue(2).tpm = {'/usr/local/spm/spm12/tpm/TPM.nii,2'};
matlabbatch{1}.spm.spatial.preproc.tissue(2).ngaus = 1;
matlabbatch{1}.spm.spatial.preproc.tissue(2).native = [0 0];
matlabbatch{1}.spm.spatial.preproc.tissue(2).warped = [1 0];
matlabbatch{1}.spm.spatial.preproc.tissue(3).tpm = {'/usr/local/spm/spm12/tpm/TPM.nii,3'};
matlabbatch{1}.spm.spatial.preproc.tissue(3).ngaus = 2;
matlabbatch{1}.spm.spatial.preproc.tissue(3).native = [0 0];
matlabbatch{1}.spm.spatial.preproc.tissue(3).warped = [1 0];
matlabbatch{1}.spm.spatial.preproc.tissue(4).tpm = {'/usr/local/spm/spm12/tpm/TPM.nii,4'};
matlabbatch{1}.spm.spatial.preproc.tissue(4).ngaus = 3;
matlabbatch{1}.spm.spatial.preproc.tissue(4).native = [0 0];
matlabbatch{1}.spm.spatial.preproc.tissue(4).warped = [0 0];
matlabbatch{1}.spm.spatial.preproc.tissue(5).tpm = {'/usr/local/spm/spm12/tpm/TPM.nii,5'};
matlabbatch{1}.spm.spatial.preproc.tissue(5).ngaus = 4;
matlabbatch{1}.spm.spatial.preproc.tissue(5).native = [0 0];
matlabbatch{1}.spm.spatial.preproc.tissue(5).warped = [0 0];
matlabbatch{1}.spm.spatial.preproc.tissue(6).tpm = {'/usr/local/spm/spm12/tpm/TPM.nii,6'};
matlabbatch{1}.spm.spatial.preproc.tissue(6).ngaus = 2;
matlabbatch{1}.spm.spatial.preproc.tissue(6).native = [0 0];
matlabbatch{1}.spm.spatial.preproc.tissue(6).warped = [0 0];
matlabbatch{1}.spm.spatial.preproc.warp.mrf = 1;
matlabbatch{1}.spm.spatial.preproc.warp.cleanup = 1;
matlabbatch{1}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
matlabbatch{1}.spm.spatial.preproc.warp.affreg = 'mni';
matlabbatch{1}.spm.spatial.preproc.warp.fwhm = 0;
matlabbatch{1}.spm.spatial.preproc.warp.samp = 3;
matlabbatch{1}.spm.spatial.preproc.warp.write = [0 1];
matlabbatch{1}.spm.spatial.preproc.warp.vox = NaN;
matlabbatch{1}.spm.spatial.preproc.warp.bb = [NaN NaN NaN
                                              NaN NaN NaN];
matlabbatch{2}.spm.spatial.normalise.write.subj.def(1) = cfg_dep('Segment: Forward Deformations', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','fordef', '()',{':'}));
matlabbatch{2}.spm.spatial.normalise.write.subj.resample(1) = cfg_dep('Segment: Bias Corrected (1)', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','channel', '()',{1}, '.','biascorr', '()',{':'}));
matlabbatch{2}.spm.spatial.normalise.write.woptions.bb = [NaN NaN NaN
                                                          NaN NaN NaN];
matlabbatch{2}.spm.spatial.normalise.write.woptions.vox = [NaN NaN NaN];
matlabbatch{2}.spm.spatial.normalise.write.woptions.interp = 4;
matlabbatch{2}.spm.spatial.normalise.write.woptions.prefix = 'w';
matlabbatch{3}.spm.util.imcalc.input(1) = cfg_dep('Segment: wc1 Images', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','tiss', '()',{1}, '.','wc', '()',{':'}));
matlabbatch{3}.spm.util.imcalc.input(2) = cfg_dep('Segment: wc2 Images', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','tiss', '()',{2}, '.','wc', '()',{':'}));
matlabbatch{3}.spm.util.imcalc.input(3) = cfg_dep('Segment: wc3 Images', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','tiss', '()',{3}, '.','wc', '()',{':'}));
matlabbatch{3}.spm.util.imcalc.input(4) = cfg_dep('Normalise: Write: Normalised Images (Subj 1)', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('()',{1}, '.','files'));
matlabbatch{3}.spm.util.imcalc.output = ['masked_brain_mri_' rid '.nii'];
matlabbatch{3}.spm.util.imcalc.outdir = {['/data2/MRI_PET_DATA/processed_images_final' suffix '/brain_stripped' suffix '/']};
matlabbatch{3}.spm.util.imcalc.expression = '((i1+i2+i3) > 0.2).*i4';
matlabbatch{3}.spm.util.imcalc.var = struct('name', {}, 'value', {});
matlabbatch{3}.spm.util.imcalc.options.dmtx = 0;
matlabbatch{3}.spm.util.imcalc.options.mask = 0;
matlabbatch{3}.spm.util.imcalc.options.interp = 1;
matlabbatch{3}.spm.util.imcalc.options.dtype = 64;
end
