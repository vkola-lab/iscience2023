%-----------------------------------------------------------------------
% Job saved on 31-May-2021 14:48:01 by cfg_util (rev $Rev: 7345 $)
% spm SPM - SPM12 (7771)
% cfg_basicio BasicIO - Unknown
%-----------------------------------------------------------------------
matlabbatch{1}.spm.util.imcalc.input = {
                                        '/home/mfromano/Research/mri-pet/metadata/mri-pet-data-raw/mni152/icbm_avg_152_t1_tal_lin.nii,1'
                                        '/home/mfromano/Research/mri-pet/metadata/mri-pet-data-raw/mni152/icbm_avg_152_t1_tal_lin_mask.nii,1'
                                        };
matlabbatch{1}.spm.util.imcalc.output = 'mni_brain.nii';
matlabbatch{1}.spm.util.imcalc.outdir = {'/home/mfromano/Research/mri-pet/metadata/mri-pet-data-raw/mni152'};
matlabbatch{1}.spm.util.imcalc.expression = 'i1.*i2';
matlabbatch{1}.spm.util.imcalc.var = struct('name', {}, 'value', {});
matlabbatch{1}.spm.util.imcalc.options.dmtx = 0;
matlabbatch{1}.spm.util.imcalc.options.mask = 0;
matlabbatch{1}.spm.util.imcalc.options.interp = 1;
matlabbatch{1}.spm.util.imcalc.options.dtype = 16;
