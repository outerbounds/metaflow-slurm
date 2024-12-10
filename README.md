# SLURM extension for Metaflow

This extension adds support for executing steps in Metaflow Flows on SLURM clusters.

## Basic Usage

- Have a SLURM cluster that you have public access for.
    - This includes the username, the IP address and the PEM file (at minimum)
- Simply add the `@slurm` decorator to the step you want to run on the SLURM cluster.

```py
@slurm(
    username="ubuntu",
    address="A.B.C.D",
    ssh_key_file="~/path/to/ssh/pem/file.pem"
)
```

Note that the above parameters can also be configured via the following environment variables:
- `METAFLOW_SLURM_USERNAME`
- `METAFLOW_SLURM_ADDRESS`
- `METAFLOW_SLURM_SSH_KEY_FILE`

The step that is decorated with `@slurm` will create the following directory structure on the cluster.

```bash
metaflow/
├── assets
│   └── madhurMovies218892mid13433160
│       └── metaflow
│           ├── INFO
│           ├── demo.py
│           ├── job.tar
│           ├── linux-64
│           ├── metaflow
│           ├── metaflow_extensions
│           └── micromamba
├── madhurMovies218892mid13433160.sh
├── stderr
│   └── madhurMovies218892mid13433160.stderr
└── stdout
    └── madhurMovies218892mid13433160.stdout
```

In the above output, `demo.py` was the name of our flow file.

One can pass `cleanup=True` in the decorator to clear up the contents of the `assets` folder.
This clears up all the `artifacts` created by Metaflow.

Using `cleanup=True` will not delete:
- `stdout` folder
- `stderr` folder
- the generated shell script i.e. `madhurMovies218892mid13433160.sh`

This is useful for debugging later and may be manually deleted by logging into the slurm cluster.

## Supplying Credentials

Credentials need to be supplied to be able to download the code package. They can:

- either exist on the Slurm cluster itself, i.e. compute instances have access to the blob store
- supplied via the `@environment` decorator

```py
@environment(vars={
    "AWS_ACCESS_KEY_ID": "XXXX",
    "AWS_SECRET_ACCESS_KEY": "YYYY"
})
```

Note that this will expose the credentials in the shell script that is generated i.e.

`madhurMovies218892mid13433160.sh` will have the following contents present:

```bash
export AWS_ACCESS_KEY_ID='XXXX'
export AWS_SECRET_ACCESS_KEY='YYYY'
```

- hydrating environment variables with the @secrets decorator from a secret manager.

PS -- If you are on the [Outerbounds](https://outerbounds.com/) platform, the auth is taken care of and there is no need to fiddle with it.

## Things to be taken care of

- The extension runs workloads via shell scripts and `sbatch` in a linux native environment
    - i.e. the workloads are NOT run inside docker containers
    - As such, the compute instances should have `python3` installed (above 3.8 preferrably)
    - If the default `python` points to `python2`, one can use the `path_to_python3` argument of the decorator i.e.

```py
@slurm(
    path_to_python3="/usr/bin/python3",
)
```

### Fin.
