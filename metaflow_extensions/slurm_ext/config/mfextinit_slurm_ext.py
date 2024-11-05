from metaflow.metaflow_config_funcs import from_conf

###
# Slurm configuration
###
# Slurm username to use with the @slurm decorator
SLURM_USERNAME = from_conf("SLURM_USERNAME")
# Slurm address to use with the @slurm decorator
SLURM_ADDRESS = from_conf("SLURM_ADDRESS")
# Slurm ssh key file to use with the @slurm decorator
SLURM_SSH_KEY_FILE = from_conf("SLURM_SSH_KEY_FILE")
# Slurm cert file to use with the @slurm decorator
SLURM_CERT_FILE = from_conf("SLURM_CERT_FILE")
# Slurm remote workdir to use with the @slurm decorator
SLURM_REMOTE_WORKDIR = from_conf("SLURM_REMOTE_WORKDIR")
