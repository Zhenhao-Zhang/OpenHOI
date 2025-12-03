
python demo/demo.py dataset=arctic \
    +test_text="[Close microwave with both hand.]" \
    +nsamples=4 \
    texthom.obj_nfeats=10 \
    hydra.output_subdir=null \
    hydra/job_logging=disabled \
    hydra/hydra_logging=disabled