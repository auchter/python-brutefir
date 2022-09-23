import argparse
import numpy as np
import os
import tempfile
import voluptuous as vol
import yaml
from pathlib import Path
from scipy import signal
from scipy.io import wavfile

TYPE_S16LE = np.dtype("<i2")


def validate_config(c, config_path=None):
    def is_path(v):
        nonlocal config_path
        if not os.path.isabs(v) and config_path is not None:
            v = os.path.join(os.path.dirname(config_path), v)
        vol.IsFile()(v)
        return v

    supported_formats = ["S16_LE", "S24_4LE"]

    io_schema = vol.Schema(
        {
            vol.Required("name"): str,
            vol.Required("channels"): int,
            vol.Required("format"): vol.In(supported_formats),
        }
    )

    filt_schema = vol.Schema(
        {
            vol.Required("name"): str,
            vol.Required("input"): str,
            vol.Required("output"): str,
            vol.Optional("coeff"): str,
        }
    )

    coeff_schema = vol.Schema(
        {
            vol.Required("name"): str,
            vol.Required("path"): vol.Any(vol.Equal("dirac pulse"), is_path),
            vol.Optional("rate"): int,
            vol.Optional("format"): vol.In(supported_formats),
        }
    )

    schema = vol.Schema(
        {
            vol.Required("sampling_rate"): int,
            vol.Required("filter_length"): vol.Any(int, str),
            vol.Optional("float_bits", default=32): vol.In([32, 64]),
            vol.Optional("cli_port"): int,
            vol.Optional("resample_coeffs", default=False): vol.Boolean(),
            "inputs": [io_schema],
            "outputs": [io_schema],
            "filters": [filt_schema],
            "coeffs": [coeff_schema],
        }
    )

    s = schema(c)
    return s


def load_yaml_config(path):
    with open(path, "r") as stream:
        y = yaml.safe_load(stream)
        return validate_config(y, config_path=path)


def load_coeff(path):
    data = None
    rate = None

    if path == "dirac pulse":
        return rate, data

    try:
        rate, data = wavfile.read(path)
    except ValueError:
        # Okay, probably not a wav file, so read raw PCM data
        # Assume mono
        fmt = TYPE_S16LE  # TODO
        data = np.fromfile(path, dtype=fmt)

    if len(data.shape) == 2:
        assert (data[:, 0] == data[:, 1]).all()
        data = data[:, 0]

    return rate, data


def load_brutefir_coeffs(config):
    coeffs = []
    for coeff in config["coeffs"]:
        rate, data = load_coeff(coeff["path"])

        if data is not None:
            if rate is not None and "rate" in coeff:
                raise RuntimeError(
                    f"cannot specify rate for wavfile in {coeff['name']}"
                )
            if rate is None:
                if "rate" not in coeff:
                    raise RuntimeError(
                        f"must specify rate for PCM data in {coeff['name']}"
                    )
                rate = coeff["rate"]

        coeffs.append(
            {
                "name": coeff["name"],
                "path": coeff["path"],
                "data": data,
                "rate": rate,
            }
        )

    return coeffs


def resample_brutefir_coeffs(config, coeffs):
    if config["resample_coeffs"] == False:
        return coeffs

    fs = config["sampling_rate"]

    for coeff in coeffs:
        if coeff["path"] == "dirac pulse":
            continue

        if coeff["rate"] == fs:
            continue

        coeff["data"] = signal.resample_poly(
            coeff["data"].astype('float64'), fs, coeff["rate"], window="hanning"
        )
        coeff["data"] = coeff["data"].astype(TYPE_S16LE)  # TODO
        coeff["rate"] = fs

    return coeffs


def gen_brutefir_coeff(config, coeff, outdir):
    # Special case for dirac pulse
    if coeff["path"] == "dirac pulse":
        return f"""
        coeff "{coeff['name']}" {{
            filename: "{coeff['path']}";
        }};
        """

    if coeff["rate"] != config["sampling_rate"]:
        raise RuntimeError(
            f"{coeff['path']} is {coeff['rate']} Hz when BruteFIR is configured for {config['sampling_rate']}. Set resample_coeffs to True to resample"
        )

    with tempfile.NamedTemporaryFile(delete=False, dir=outdir) as f:
        assert coeff["data"].dtype == TYPE_S16LE
        coeff["data"].tofile(f.name)
        filename = Path(f.name).name
        return f"""
        coeff "{coeff['name']}" {{
            filename: "{filename}";
            format: "S16_LE";
            blocks: -1;
        }};
        """


def gen_brutefir_io(c, io, block):
    names = [io["name"] + "_" + str(x) for x in range(io["channels"])]
    names = ", ".join([f'"{name}"' for name in names])
    path = "/dev/stdout" if block == "output" else "/dev/stdin"
    chans = str(io["channels"]) + "/" + ",".join(map(str, range(io["channels"])))
    mute = ",".join(["false" for _ in range(io["channels"])])
    delay = ",".join(["0" for _ in range(io["channels"])])

    return f"""
    {block} {names} {{
        device: "file" {{ path: "{path}"; }};
        sample: "{io['format']}";
        channels: {chans};
        delay: {delay};
        maxdelay: -1;
        mute: {mute};
    }};
    """


def gen_brutefir_filter(c, f):
    in_chans = 0
    for i in c["inputs"]:
        if i["name"] == f["input"]:
            in_chans = i["channels"]

    out_chans = 0
    for o in c["outputs"]:
        if o["name"] == f["output"]:
            out_chans = o["channels"]

    if in_chans == 0 or out_chans == 0:
        raise RuntimeError("Unknown channels...")
    if in_chans != out_chans:
        raise RuntimeError(
            f"Input and output channels for filter {f['name']} must match!"
        )

    coeff = -1
    if "coeff" in f:
        coeff = f'"{f["coeff"]}"'

    s = ""
    for i in range(in_chans):
        s += f"""
        filter "{f['name']}_{i}" {{
            from_inputs: "{f['input']}_{i}"/0.0;
            to_outputs: "{f['output']}_{i}"/0.0;
            process: -1;
            coeff: {coeff};
            delay: 0;
            crossfade: false;
        }};
        """

    return s


def generate_brutefir_config(config, coeffs, outdir):
    s = ""
    s += """
    overflow_warnings: true;
    show_progress: false;
    max_dither_table_size: 0;
    allow_poll_mode: false;
    modules_path: ".";
    monitor_rate: false;
    powersave: true;
    lock_memory: true;
    sdf_length: -1;
    safety_limit: 20;
    convolver_config: "~/.brutefir_convolver";

    """

    s += f"""
    float_bits: {config['float_bits']};
    sampling_rate: {config['sampling_rate']};
    filter_length: {config['filter_length']};

    """

    if "cli_port" in config:
        s += f'logic: "cli" {{ port: {config["cli_port"]}; }};'

    for coeff in coeffs:
        s += gen_brutefir_coeff(config, coeff, outdir)

    for i in config["inputs"]:
        s += gen_brutefir_io(config, i, "input")

    for o in config["outputs"]:
        s += gen_brutefir_io(config, o, "output")

    for f in config["filters"]:
        s += gen_brutefir_filter(config, f)

    cfgfile = Path(outdir) / "brutefir_config"
    cfgfile.write_text(s)

    return cfgfile


def gen_config():
    parser = argparse.ArgumentParser(description="Generate BruteFIR config")
    parser.add_argument("config", type=str, help="configuration input")
    parser.add_argument("--output", type=str, help="output directory")
    args = parser.parse_args()

    if args.output is not None:
        outdir = Path(args.output)
        outdir.mkdir(parents=True, exist_ok=True)
    else:
        outdir = Path(tempfile.mkdtemp())

    config = load_yaml_config(args.config)

    coeffs = load_brutefir_coeffs(config)
    coeffs = resample_brutefir_coeffs(config, coeffs)

    cfgfile = generate_brutefir_config(config, coeffs, outdir=outdir)

    print(f"Generated config written to {outdir}")
