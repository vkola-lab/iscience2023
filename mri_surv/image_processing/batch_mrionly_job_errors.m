function batch_mrionly_job_errors(suffix)
    addpath(genpath('/usr/local/spm/spm12/'));
    BASE_DIR = ['/data2/MRI_PET_DATA/processed_images_final' suffix filesep];
    MRI_folder = [BASE_DIR 'ADNI_MRI_nii_recenter_NL' suffix filesep];

    maxNumCompThreads = 1;
    spm('defaults', 'PET');
    spm_jobman('initcfg');
    rand('state', 10);
    newdir = ['/data2/MRI_PET_DATA/processed_images_final' suffix '/ADNI_MRI_nii_recentered_cat12' suffix];
    mkdir(newdir);

    fnames = dir(MRI_folder);
    fnames = {fnames.name};

    disp(fnames)

    parpool(10);

    fi = fopen('logs/error_files.txt','w');

    parfor i=1:length(fnames)
        if fnames{i} == '.' or fnames{i} == '..'
            continue
        end
        if exist([newdir filesep fnames{i}],'file') == 2
            disp(['skipping ' newdir filesep fnames{i}])
            continue
        end
        try
            system(['rsync -av ' MRI_folder  fnames{i} ' ' newdir filesep]);
            rand('state', 10);
            jobs = process_individual_rid(fnames{i}, newdir);
            cat12('expert')
            spm('defaults', 'PET');
            spm_jobman('initcfg');
            spm_jobman('run_nogui', jobs);
        catch
            fprintf(fi,['skipping ' fnames{i} '\n'])
            disp(['skipping ' fnames{i}])
        end
    end
end

function matlabbatch = process_individual_rid(fname, newdir)
    matlabbatch{1}.cfg_basicio.file_dir.file_ops.file_fplist.dir = {newdir};
    matlabbatch{1}.cfg_basicio.file_dir.file_ops.file_fplist.filter = fname;
    matlabbatch{1}.cfg_basicio.file_dir.file_ops.file_fplist.rec = 'FPList';
    matlabbatch{2}.spm.tools.cat.estwrite.data(1) = cfg_dep('File Selector (Batch Mode): Selected Files (^0974_mri.nii$)', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','files'));
    matlabbatch{2}.spm.tools.cat.estwrite.data_wmh = {''};
    matlabbatch{2}.spm.tools.cat.estwrite.nproc = 1;
    matlabbatch{2}.spm.tools.cat.estwrite.useprior = '';
    matlabbatch{2}.spm.tools.cat.estwrite.opts.tpm = {'/usr/local/spm/spm12/tpm/TPM.nii'};
    matlabbatch{2}.spm.tools.cat.estwrite.opts.affreg = 'mni';
    matlabbatch{2}.spm.tools.cat.estwrite.opts.biasstr = 0.5;
    matlabbatch{2}.spm.tools.cat.estwrite.opts.accstr = 0.5;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.segmentation.APP = 1070;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.segmentation.NCstr = -Inf;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.segmentation.spm_kamap = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.segmentation.LASstr = 0.5;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.segmentation.gcutstr = 2;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.segmentation.cleanupstr = 0.5;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.segmentation.BVCstr = 0.5;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.segmentation.WMHC = 1;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.segmentation.SLC = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.segmentation.mrf = 1;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.segmentation.restypes.optimal = [1 0.1];
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.registration.shooting.shootingtpm = {'/usr/local/spm/spm12/toolbox/cat12/templates_volumes/Template_0_IXI555_MNI152_GS.nii'};
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.registration.shooting.regstr = 0.5;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.vox = 1.5;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.surface.pbtres = 0.5;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.surface.pbtmethod = 'pbt2x';
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.surface.pbtlas = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.surface.collcorr = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.surface.reduce_mesh = 1;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.surface.vdist = 1.33333333333333;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.surface.scale_cortex = 0.7;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.surface.add_parahipp = 0.1;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.surface.close_parahipp = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.admin.experimental = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.admin.new_release = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.admin.lazy = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.admin.ignoreErrors = 1;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.admin.verb = 2;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.admin.print = 2;
    matlabbatch{2}.spm.tools.cat.estwrite.output.surface = 1;
    matlabbatch{2}.spm.tools.cat.estwrite.output.surf_measures = 1;
    matlabbatch{2}.spm.tools.cat.estwrite.output.ROImenu.atlases.neuromorphometrics = 1;
    matlabbatch{2}.spm.tools.cat.estwrite.output.ROImenu.atlases.lpba40 = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.ROImenu.atlases.cobra = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.ROImenu.atlases.hammers = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.ROImenu.atlases.ibsr = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.ROImenu.atlases.aal3 = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.ROImenu.atlases.mori = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.ROImenu.atlases.anatomy = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.ROImenu.atlases.julichbrain = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.ROImenu.atlases.Schaefer2018_100Parcels_17Networks_order = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.ROImenu.atlases.Schaefer2018_200Parcels_17Networks_order = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.ROImenu.atlases.Schaefer2018_400Parcels_17Networks_order = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.ROImenu.atlases.Schaefer2018_600Parcels_17Networks_order = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.ROImenu.atlases.ownatlas = {''};
    matlabbatch{2}.spm.tools.cat.estwrite.output.GM.native = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.GM.warped = 1;
    matlabbatch{2}.spm.tools.cat.estwrite.output.GM.mod = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.GM.dartel = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.WM.native = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.WM.warped = 1;
    matlabbatch{2}.spm.tools.cat.estwrite.output.WM.mod = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.WM.dartel = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.CSF.native = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.CSF.warped = 1;
    matlabbatch{2}.spm.tools.cat.estwrite.output.CSF.mod = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.CSF.dartel = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.ct.native = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.ct.warped = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.ct.dartel = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.pp.native = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.pp.warped = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.pp.dartel = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.WMH.native = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.WMH.warped = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.WMH.mod = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.WMH.dartel = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.SL.native = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.SL.warped = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.SL.mod = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.SL.dartel = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.TPMC.native = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.TPMC.warped = 1;
    matlabbatch{2}.spm.tools.cat.estwrite.output.TPMC.mod = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.TPMC.dartel = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.atlas.native = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.atlas.warped = 1;
    matlabbatch{2}.spm.tools.cat.estwrite.output.atlas.dartel = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.label.native = 1;
    matlabbatch{2}.spm.tools.cat.estwrite.output.label.warped = 1;
    matlabbatch{2}.spm.tools.cat.estwrite.output.label.dartel = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.labelnative = 1;
    matlabbatch{2}.spm.tools.cat.estwrite.output.bias.native = 1;
    matlabbatch{2}.spm.tools.cat.estwrite.output.bias.warped = 1;
    matlabbatch{2}.spm.tools.cat.estwrite.output.bias.dartel = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.las.native = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.las.warped = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.las.dartel = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.jacobianwarped = 1;
    matlabbatch{2}.spm.tools.cat.estwrite.output.warps = [1 1];
    matlabbatch{2}.spm.tools.cat.estwrite.output.rmat = 1;
end