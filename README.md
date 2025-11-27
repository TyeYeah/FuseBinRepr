# FuseBinRepr

## Usage

All in one:
```sh
$ python run_fuse.py --expmode cmdrun 
```

Training example. With `expmode` set to `train`, the folder and size are aligned with the corresponding foundation embedding model.
```sh
$ python run_fuse.py --note clap_gmn_self_all --expmode train --epoch-num 10 --todevice=0 --train-text-folder /path/to/clap_cache/out/ --train-graph-folder /path/to/gmn_cache/out/ --text-size 768 --graph-size 1024
```

Testing example: With `expmode` set to `eval`, the configuration also includes a pretrained model path.
```sh
$ python run_fuse.py --note clap_gmn_self_all --expmode eval --pretrained-path /path/to/model/clap_gmn_self_all_model.ep0.pt --train-text-folder /path/to/clap_cache/out/ --train-graph-folder /path/to/gmn_cache/out/ --text-size 768 --graph-size 1024
```