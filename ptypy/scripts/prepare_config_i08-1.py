import os
import pathlib
from argparse import ArgumentParser, Namespace
from collections.abc import Sequence

import httpx
import yaml
from tiled.client import from_uri


# No-op auth to prevent separate login
class WorkflowsAuth(httpx.Auth):
    def auth_flow(self, request):
        yield request


def parse(args: Sequence[str] | None = None) -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--scan_id")
    parser.add_argument("--visit_dir", type=pathlib.Path)
    parser.add_argument("--outpath", type=pathlib.Path)
    parser.add_argument("--numiter", type=int)
    parser.add_argument("--probe_modes", type=int)
    parser.add_argument("--binning", type=int)
    parser.add_argument("--defocus_microns", type=int)
    parser.add_argument("--usegpu", type=bool)
    return parser.parse_args(args)


def main(args: Sequence[str] | None = None) -> None:
    params = parse(args)

    instrument_session = str(params.visit_dir).split("/")[-1]
    proposal_code = instrument_session.split("-")[0]

    # client = from_uri(os.environ["TILED_URL"], auth=WorkflowsAuth())
    client = from_uri("https://tiled.diamond.ac.uk")

    scan = client["i08-1"][proposal_code][instrument_session][params.scan_id]["primary"]

    scan_number = client["i08-1"][proposal_code][instrument_session][
        params.scan_id
    ].metadata["start"]["scan_id"]

    with open("/tmp/scan_number.txt", "w") as w:
        w.write(str(scan_number))

    AndorMarana = scan["AndorMarana"].data_sources()[0]
    intensities_key = AndorMarana.parameters["dataset"]
    intensities_file = AndorMarana.assets[0].data_uri.removeprefix("file://localhost")

    SampleX = scan["SampleX"].data_sources()[0]
    positions_file = SampleX.assets[0].data_uri.removeprefix("file://localhost")
    fast_key = SampleX.parameters["dataset"]

    SampleY = scan["SampleY"].data_sources()[0]
    slow_key = SampleY.parameters["dataset"]

    with open("/tmp/ptypy_config.yaml", "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    data["parameter_tree"]["engines"]["engine0"]["numiter"] = params.numiter
    data["parameter_tree"]["engines"]["engine0"]["name"] = (
        "RAAR_cupy" if params.usegpu else "RAAR"
    )

    data_scan = data["parameter_tree"]["scans"]["scan_00"]

    data_scan["illumination"]["propagation"]["parallel"] = (
        params.defocus_microns / 10**6
    )
    data_scan["data"]["rebin"] = params.binning

    data_scan["data"]["intensities"]["file"] = intensities_file
    data_scan["data"]["intensities"]["key"] = intensities_key

    scan_positions = data_scan["data"]["positions"]
    scan_positions["file"] = positions_file
    scan_positions["fast_key"] = fast_key
    scan_positions["slow_key"] = slow_key

    data_scan["coherence"]["num_probe_modes"] = params.probe_modes

    config_path = os.path.join(
        params.visit_dir, params.outpath, f"scan_{scan_number}", "python_config.yaml"
    )

    with open("/tmp/config_path.txt", "w") as w:
        w.write(config_path)

    with open(config_path, "w") as o:
        yaml.dump(data, o, sort_keys=False)


if __name__ == "__main__":
    main()
