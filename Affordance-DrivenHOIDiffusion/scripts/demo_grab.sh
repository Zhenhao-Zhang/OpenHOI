python demo/demo.py dataset=grab \
    +test_text="[Chop knife with right hand.]" \
    +nsamples=4 \
    hydra.output_subdir=null \
    hydra/job_logging=disabled \
    hydra/hydra_logging=disabled