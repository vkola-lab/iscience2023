function process_files(suffix)
addpath(genpath('/home/mfromano/spm/spm12/'));
BASE_DIR = ['/data2/MRI_PET_DATA/processed_images_final' suffix filesep];
MRI_folder_new = [BASE_DIR 'ADNI_MRI_nii_recenter' suffix filesep];

fnames = dir(MRI_folder_new);
fnames = {fnames.name};
rids = arrayfun(@(x) regexp(x,'^([0-9]{4}).*\.nii$','tokens'), fnames, 'uniformoutput', false);
rids = [rids{:}];
rids = [rids{:}];
rids = [rids{:}];
mkdir(['/data2/MRI_PET_DATA/processed_images_final' suffix filesep 'ADNI_MRI_nii_recenter_amyloid' suffix]);
mkdir(['/data2/MRI_PET_DATA/processed_images_final' suffix filesep 'brain_stripped' suffix]);
rand('state',10);
maxNumCompThreads = 1;
spm('defaults', 'PET');
spm_jobman('initcfg');
parpool(14);
parfor i=1:length(rids)
        rand('state',10);
        curr_mri = ['/data2/MRI_PET_DATA/processed_images_final' suffix '/ADNI_MRI_nii_recenter' suffix filesep rids{i} '_mri.nii'];
        disp(['copying ' rids{i}])
        system(['rsync -av ' curr_mri ' /data2/MRI_PET_DATA/processed_images_final' suffix '/ADNI_MRI_nii_recenter_amyloid' suffix]);
%         system(['rsync -av ' curr_mri ' /data2/MRI_PET_DATA/processed_images_final' suffix '/ADNI_MRI_nii_recenter_fdg' suffix filesep]);
        jobs = batch_process_amyloidmri(rids{i}, suffix);
        spm_jobman('run_nogui', jobs);
end
% 
% spm('defaults', 'PET');
% spm_jobman('initcfg');
% 
% parfor i=1:length(rids)
%         rand('state',10);
%         jobs = batch_process_fdg(rids{i}, suffix);
%         job = spm_jobman('run_nogui', jobs);
% end

% modality = {'amyloid','mri','fdg'};
modality = {'amyloid','mri'};
for i=1:length(modality)
modal = modality{i};
cdir = ['/data2/MRI_PET_DATA/processed_images_final' suffix filesep 'brain_stripped_' modal suffix];
mkdir(cdir);
system(['rsync -ar /data2/MRI_PET_DATA/processed_images_final' suffix filesep 'brain_stripped' suffix filesep '*' modal '.nii ' cdir filesep]);
disp(['rsync -ar /data2/MRI_PET_DATA/processed_images_final' suffix filesep 'brain_stripped' suffix filesep '*' modal '.nii ' cdir filesep])
end

end
%-----------------------------------------------------------------------
% Job saved on 22-Oct-2020 12:09:13 by cfg_util (rev $Rev: 7345 $)
% spm SPM - SPM12 (7771)
% cfg_basicio BasicIO - Unknown
%-----------------------------------------------------------------------
function matlabbatch = batch_process_amyloidmri(rid, suffix)
matlabbatch{1}.spm.spatial.coreg.estimate.ref = {['/data2/MRI_PET_DATA/processed_images_final' suffix '/ADNI_AMYLOID_nii_recenter' suffix '/' rid '_amyloid.nii,1']};
matlabbatch{1}.spm.spatial.coreg.estimate.source = {['/data2/MRI_PET_DATA/processed_images_final' suffix '/ADNI_MRI_nii_recenter_amyloid' suffix '/' rid '_mri.nii,1']};
matlabbatch{1}.spm.spatial.coreg.estimate.other = {''};
matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.cost_fun = 'nmi';
matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.sep = [4 2];
matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.tol = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.fwhm = [7 7];
matlabbatch{2}.spm.spatial.preproc.channel.vols(1) = cfg_dep('Coregister: Estimate: Coregistered Images', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','cfiles'));
matlabbatch{2}.spm.spatial.preproc.channel.biasreg = 0.001;
matlabbatch{2}.spm.spatial.preproc.channel.biasfwhm = 60;
matlabbatch{2}.spm.spatial.preproc.channel.write = [0 1];
matlabbatch{2}.spm.spatial.preproc.tissue(1).tpm = {'/usr/local/spm/spm12/tpm/TPM.nii,1'};
matlabbatch{2}.spm.spatial.preproc.tissue(1).ngaus = 1;
matlabbatch{2}.spm.spatial.preproc.tissue(1).native = [1 1];
matlabbatch{2}.spm.spatial.preproc.tissue(1).warped = [1 0];
matlabbatch{2}.spm.spatial.preproc.tissue(2).tpm = {'/usr/local/spm/spm12/tpm/TPM.nii,2'};
matlabbatch{2}.spm.spatial.preproc.tissue(2).ngaus = 1;
matlabbatch{2}.spm.spatial.preproc.tissue(2).native = [1 1];
matlabbatch{2}.spm.spatial.preproc.tissue(2).warped = [1 0];
matlabbatch{2}.spm.spatial.preproc.tissue(3).tpm = {'/usr/local/spm/spm12/tpm/TPM.nii,3'};
matlabbatch{2}.spm.spatial.preproc.tissue(3).ngaus = 2;
matlabbatch{2}.spm.spatial.preproc.tissue(3).native = [1 1];
matlabbatch{2}.spm.spatial.preproc.tissue(3).warped = [1 0];
matlabbatch{2}.spm.spatial.preproc.tissue(4).tpm = {'/usr/local/spm/spm12/tpm/TPM.nii,4'};
matlabbatch{2}.spm.spatial.preproc.tissue(4).ngaus = 3;
matlabbatch{2}.spm.spatial.preproc.tissue(4).native = [1 1];
matlabbatch{2}.spm.spatial.preproc.tissue(4).warped = [1 0];
matlabbatch{2}.spm.spatial.preproc.tissue(5).tpm = {'/usr/local/spm/spm12/tpm/TPM.nii,5'};
matlabbatch{2}.spm.spatial.preproc.tissue(5).ngaus = 4;
matlabbatch{2}.spm.spatial.preproc.tissue(5).native = [0 0];
matlabbatch{2}.spm.spatial.preproc.tissue(5).warped = [0 1];
matlabbatch{2}.spm.spatial.preproc.tissue(6).tpm = {'/usr/local/spm/spm12/tpm/TPM.nii,6'};
matlabbatch{2}.spm.spatial.preproc.tissue(6).ngaus = 2;
matlabbatch{2}.spm.spatial.preproc.tissue(6).native = [0 0];
matlabbatch{2}.spm.spatial.preproc.tissue(6).warped = [0 0];
matlabbatch{2}.spm.spatial.preproc.warp.mrf = 1;
matlabbatch{2}.spm.spatial.preproc.warp.cleanup = 0;
matlabbatch{2}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
matlabbatch{2}.spm.spatial.preproc.warp.affreg = 'mni';
matlabbatch{2}.spm.spatial.preproc.warp.fwhm = 0;
matlabbatch{2}.spm.spatial.preproc.warp.samp = 2;
matlabbatch{2}.spm.spatial.preproc.warp.write = [1 1];
matlabbatch{2}.spm.spatial.preproc.warp.vox = NaN;
matlabbatch{2}.spm.spatial.preproc.warp.bb = [NaN NaN NaN
                                              NaN NaN NaN];
matlabbatch{3}.spm.spatial.normalise.write.subj(1).def(1) = cfg_dep('Segment: Forward Deformations', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','fordef', '()',{':'}));
matlabbatch{3}.spm.spatial.normalise.write.subj(1).resample(1) = cfg_dep('Segment: Bias Corrected (1)', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','channel', '()',{1}, '.','biascorr', '()',{':'}));
matlabbatch{3}.spm.spatial.normalise.write.subj(2).def(1) = cfg_dep('Segment: Forward Deformations', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','fordef', '()',{':'}));
matlabbatch{3}.spm.spatial.normalise.write.subj(2).resample = {['/data2/MRI_PET_DATA/processed_images_final' suffix '/ADNI_AMYLOID_nii_recenter' suffix '/' rid '_amyloid.nii,1']};
matlabbatch{3}.spm.spatial.normalise.write.woptions.bb = [NaN NaN NaN
                                                          NaN NaN NaN];
matlabbatch{3}.spm.spatial.normalise.write.woptions.vox = [1.5 1.5 1.5];
matlabbatch{3}.spm.spatial.normalise.write.woptions.interp = 4;
matlabbatch{3}.spm.spatial.normalise.write.woptions.prefix = 'w';
matlabbatch{4}.spm.util.imcalc.input(1) = cfg_dep('Segment: wc1 Images', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','tiss', '()',{1}, '.','wc', '()',{':'}));
matlabbatch{4}.spm.util.imcalc.input(2) = cfg_dep('Segment: wc2 Images', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','tiss', '()',{2}, '.','wc', '()',{':'}));
matlabbatch{4}.spm.util.imcalc.input(3) = cfg_dep('Segment: wc3 Images', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','tiss', '()',{3}, '.','wc', '()',{':'}));
matlabbatch{4}.spm.util.imcalc.output = [rid '_brainmask_amyloidmri'];
matlabbatch{4}.spm.util.imcalc.outdir = {''};
matlabbatch{4}.spm.util.imcalc.expression = '(i1+i2+i3) > .1';
matlabbatch{4}.spm.util.imcalc.var = struct('name', {}, 'value', {});
matlabbatch{4}.spm.util.imcalc.options.dmtx = 0;
matlabbatch{4}.spm.util.imcalc.options.mask = 0;
matlabbatch{4}.spm.util.imcalc.options.interp = 1;
matlabbatch{4}.spm.util.imcalc.options.dtype = 768;
matlabbatch{5}.spm.util.imcalc.input(1) = cfg_dep('Normalise: Write: Normalised Images (Subj 1)', substruct('.','val', '{}',{3}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('()',{1}, '.','files'));
matlabbatch{5}.spm.util.imcalc.input(2) = cfg_dep('Image Calculator: ImCalc Computed Image: brainmask_amyloidmri', substruct('.','val', '{}',{4}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','files'));
matlabbatch{5}.spm.util.imcalc.output = [rid '_brain_mri'];
matlabbatch{5}.spm.util.imcalc.outdir = {['/data2/MRI_PET_DATA/processed_images_final' suffix '/brain_stripped' suffix '/']};
matlabbatch{5}.spm.util.imcalc.expression = 'i1.*i2';
matlabbatch{5}.spm.util.imcalc.var = struct('name', {}, 'value', {});
matlabbatch{5}.spm.util.imcalc.options.dmtx = 0;
matlabbatch{5}.spm.util.imcalc.options.mask = 0;
matlabbatch{5}.spm.util.imcalc.options.interp = 1;
matlabbatch{5}.spm.util.imcalc.options.dtype = 64;
matlabbatch{6}.spm.util.imcalc.input(1) = cfg_dep('Normalise: Write: Normalised Images (Subj 2)', substruct('.','val', '{}',{3}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('()',{2}, '.','files'));
matlabbatch{6}.spm.util.imcalc.input(2) = cfg_dep('Image Calculator: ImCalc Computed Image: brainmask_amyloidmri', substruct('.','val', '{}',{4}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','files'));
matlabbatch{6}.spm.util.imcalc.output = [rid '_brain_amyloid'];
matlabbatch{6}.spm.util.imcalc.outdir = {['/data2/MRI_PET_DATA/processed_images_final' suffix '/brain_stripped' suffix '/']};
matlabbatch{6}.spm.util.imcalc.expression = 'i1.*i2';
matlabbatch{6}.spm.util.imcalc.var = struct('name', {}, 'value', {});
matlabbatch{6}.spm.util.imcalc.options.dmtx = 0;
matlabbatch{6}.spm.util.imcalc.options.mask = 0;
matlabbatch{6}.spm.util.imcalc.options.interp = 1;
matlabbatch{6}.spm.util.imcalc.options.dtype = 64;
end

% function matlabbatch = batch_process_fdg(rid, suffix)
% matlabbatch{1}.spm.spatial.coreg.estimate.ref = {['/data2/MRI_PET_DATA/processed_images_final' suffix '/ADNI_FDG_nii_recenter' suffix '/' rid '_fdg.nii,1']};
% matlabbatch{1}.spm.spatial.coreg.estimate.source = {['/data2/MRI_PET_DATA/processed_images_final' suffix '/ADNI_MRI_nii_recenter_fdg' suffix '/' rid '_mri.nii,1']};
% matlabbatch{1}.spm.spatial.coreg.estimate.other = {''};
% matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.cost_fun = 'nmi';
% matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.sep = [4 2];
% matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.tol = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
% matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.fwhm = [7 7];
% matlabbatch{2}.spm.spatial.preproc.channel.vols(1) = cfg_dep('Coregister: Estimate: Coregistered Images', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','cfiles'));
% matlabbatch{2}.spm.spatial.preproc.channel.biasreg = 0.001;
% matlabbatch{2}.spm.spatial.preproc.channel.biasfwhm = 60;
% matlabbatch{2}.spm.spatial.preproc.channel.write = [0 1];
% matlabbatch{2}.spm.spatial.preproc.tissue(1).tpm = {'/usr/local/spm/spm12/tpm/TPM.nii,1'};
% matlabbatch{2}.spm.spatial.preproc.tissue(1).ngaus = 1;
% matlabbatch{2}.spm.spatial.preproc.tissue(1).native = [1 1];
% matlabbatch{2}.spm.spatial.preproc.tissue(1).warped = [1 0];
% matlabbatch{2}.spm.spatial.preproc.tissue(2).tpm = {'/usr/local/spm/spm12/tpm/TPM.nii,2'};
% matlabbatch{2}.spm.spatial.preproc.tissue(2).ngaus = 1;
% matlabbatch{2}.spm.spatial.preproc.tissue(2).native = [1 1];
% matlabbatch{2}.spm.spatial.preproc.tissue(2).warped = [1 0];
% matlabbatch{2}.spm.spatial.preproc.tissue(3).tpm = {'/usr/local/spm/spm12/tpm/TPM.nii,3'};
% matlabbatch{2}.spm.spatial.preproc.tissue(3).ngaus = 2;
% matlabbatch{2}.spm.spatial.preproc.tissue(3).native = [1 1];
% matlabbatch{2}.spm.spatial.preproc.tissue(3).warped = [1 0];
% matlabbatch{2}.spm.spatial.preproc.tissue(4).tpm = {'/usr/local/spm/spm12/tpm/TPM.nii,4'};
% matlabbatch{2}.spm.spatial.preproc.tissue(4).ngaus = 3;
% matlabbatch{2}.spm.spatial.preproc.tissue(4).native = [1 1];
% matlabbatch{2}.spm.spatial.preproc.tissue(4).warped = [1 0];
% matlabbatch{2}.spm.spatial.preproc.tissue(5).tpm = {'/usr/local/spm/spm12/tpm/TPM.nii,5'};
% matlabbatch{2}.spm.spatial.preproc.tissue(5).ngaus = 4;
% matlabbatch{2}.spm.spatial.preproc.tissue(5).native = [0 0];
% matlabbatch{2}.spm.spatial.preproc.tissue(5).warped = [1 0];
% matlabbatch{2}.spm.spatial.preproc.tissue(6).tpm = {'/usr/local/spm/spm12/tpm/TPM.nii,6'};
% matlabbatch{2}.spm.spatial.preproc.tissue(6).ngaus = 2;
% matlabbatch{2}.spm.spatial.preproc.tissue(6).native = [0 0];
% matlabbatch{2}.spm.spatial.preproc.tissue(6).warped = [0 0];
% matlabbatch{2}.spm.spatial.preproc.warp.mrf = 1;
% matlabbatch{2}.spm.spatial.preproc.warp.cleanup = 0;
% matlabbatch{2}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
% matlabbatch{2}.spm.spatial.preproc.warp.affreg = 'mni';
% matlabbatch{2}.spm.spatial.preproc.warp.fwhm = 0;
% matlabbatch{2}.spm.spatial.preproc.warp.samp = 2;
% matlabbatch{2}.spm.spatial.preproc.warp.write = [1 1];
% matlabbatch{2}.spm.spatial.preproc.warp.vox = NaN;
% matlabbatch{2}.spm.spatial.preproc.warp.bb = [NaN NaN NaN
%                                               NaN NaN NaN];
% matlabbatch{3}.spm.spatial.normalise.write.subj.def(1) = cfg_dep('Segment: Forward Deformations', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','fordef', '()',{':'}));
% matlabbatch{3}.spm.spatial.normalise.write.subj.resample = {['/data2/MRI_PET_DATA/processed_images_final' suffix '/ADNI_FDG_nii_recenter' suffix '/' rid '_fdg.nii,1']};
% matlabbatch{3}.spm.spatial.normalise.write.woptions.bb = [NaN NaN NaN
%                                                           NaN NaN NaN];
% matlabbatch{3}.spm.spatial.normalise.write.woptions.vox = [1.5 1.5 1.5];
% matlabbatch{3}.spm.spatial.normalise.write.woptions.interp = 4;
% matlabbatch{3}.spm.spatial.normalise.write.woptions.prefix = 'w';
% matlabbatch{4}.spm.util.imcalc.input(1) = cfg_dep('Segment: wc1 Images', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','tiss', '()',{1}, '.','wc', '()',{':'}));
% matlabbatch{4}.spm.util.imcalc.input(2) = cfg_dep('Segment: wc2 Images', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','tiss', '()',{2}, '.','wc', '()',{':'}));
% matlabbatch{4}.spm.util.imcalc.input(3) = cfg_dep('Segment: wc3 Images', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','tiss', '()',{3}, '.','wc', '()',{':'}));
% matlabbatch{4}.spm.util.imcalc.output = [rid '_brainmask_fdg'];
% matlabbatch{4}.spm.util.imcalc.outdir = {''};
% matlabbatch{4}.spm.util.imcalc.expression = '(i1+i2+i3) > 0.1';
% matlabbatch{4}.spm.util.imcalc.var = struct('name', {}, 'value', {});
% matlabbatch{4}.spm.util.imcalc.options.dmtx = 0;
% matlabbatch{4}.spm.util.imcalc.options.mask = 0;
% matlabbatch{4}.spm.util.imcalc.options.interp = 1;
% matlabbatch{4}.spm.util.imcalc.options.dtype = 768;
% matlabbatch{5}.spm.util.imcalc.input(1) = cfg_dep('Normalise: Write: Normalised Images (Subj 1)', substruct('.','val', '{}',{3}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('()',{1}, '.','files'));
% matlabbatch{5}.spm.util.imcalc.input(2) = cfg_dep('Image Calculator: ImCalc Computed Image: brainmask_fdg', substruct('.','val', '{}',{4}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','files'));
% matlabbatch{5}.spm.util.imcalc.output = [rid '_brain_fdg'];
% matlabbatch{5}.spm.util.imcalc.outdir = {['/data2/MRI_PET_DATA/processed_images_final' suffix '/brain_stripped' suffix '/']};
% matlabbatch{5}.spm.util.imcalc.expression = 'i1.*i2';
% matlabbatch{5}.spm.util.imcalc.var = struct('name', {}, 'value', {});
% matlabbatch{5}.spm.util.imcalc.options.dmtx = 0;
% matlabbatch{5}.spm.util.imcalc.options.mask = 0;
% matlabbatch{5}.spm.util.imcalc.options.interp = 1;
% matlabbatch{5}.spm.util.imcalc.options.dtype = 64;
% end
