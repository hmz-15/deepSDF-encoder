import json
import os
import argparse

import numpy as np


def arg_parser():
    """Argument Parser
    Parse arguments from command line, and perform error checking
    Returns:
        An argument object which contains arguments from cmd line
    """
    parser = argparse.ArgumentParser(prog='Dataset Unique Mesh Filter')

    parser.add_argument(
        "-s"
        "--src",
        dest="src",
        type=str,
        required=True,
        help="Source file that map mesh filename to ID"
    )
    parser.add_argument(
        "-l"
        "--latent",
        dest="latent",
        type=str,
        required=True,
        help="Latent vector file"
    )
    parser.add_argument(
        "-o"
        "--out",
        dest="out",
        type=str,
        required=True,
        help="Output directory"
    )

    return parser.parse_args()


def load_json(filename):
    data = None

    with open(filename, "r") as fin:
        data = json.load(fin)

    return data


def meshfilename_to_latent(source_file, latent_file, output_file):
    source_data = load_json(source_file)

    # {
    #     "id": {"latent": [xx, xx, ...], "scale": xx}
    # }
    latent_json = load_json(latent_file)
    
    id2filename = {
        it[1]: it[0].split('/')[-1].split('.')[0] for it in source_data
    }

    filename2latent = {
        id2filename[int(k)]: v['latent'] for k, v in latent_json.items()
    }

    data = {
        "mesh_source": "/".join(source_data[0][0].split('/')[:-1]),
        "latent": filename2latent
    }
    
    with open(output_file, "w") as fout:
        fout.write( json.dumps(data, indent=4) )


if __name__ == "__main__":
    args = arg_parser()
    meshfilename_to_latent(args.src, args.latent, args.out)