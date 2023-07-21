% List of open inputs
nrun = X; % enter the number of runs here
jobfile = {'/home/mfromano/Research/mri-pet/image_processing/mask_mni_brain_job.m'};
jobs = repmat(jobfile, 1, nrun);
inputs = cell(0, nrun);
for crun = 1:nrun
end
spm('defaults', 'PET');
spm_jobman('run', jobs, inputs{:});
