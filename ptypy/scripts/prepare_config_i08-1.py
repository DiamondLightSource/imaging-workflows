import os
import yaml
import pathlib
import httpx
from argparse import ArgumentParser, Namespace
from collections.abc import Sequence

# No-op auth to prevent separate login
class WorkflowsAuth(httpx.Auth):
  def auth_flow(self, request):
    yield request

def parse(args: Sequence[str] | None = None) -> Namespace:
  parser = ArgumentParser()
  parser.add_argument("--inpath", type=pathlib.Path)
  parser.add_argument("--numiter", type=int)
  parser.add_argument("--probe_modes", type=int)
  parser.add_argument("--binning", type=int)
  parser.add_argument("--defocus_microns", type=int)
  return parser.parse_args(args)

def main(args: Sequence[str] | None = None) -> None:
  params = parse(args)
  print(params)

  print(os.environ)
  with open("ptypy_config.yaml", "r") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)

  data["visit_path"] = str(params.inpath)
  data["parameter_tree"]["engines"]["engine0"]["numiter"] = params.numiter
  data["parameter_tree"]["scans"]["scan_00"]["illumination"]["propagation"]["parallel"] = params.defocus_microns/10**6
  data["parameter_tree"]["scans"]["scan_00"]["data"]["rebin"] = params.binning
  data["parameter_tree"]["scans"]["scan_00"]["coherence"]["num_probe_modes"] = params.probe_modes

  with open("ptypy_config.yaml", "w") as o:    
    yaml.dump(data, o, sort_keys=False)

if __name__ == "__main__":
  main()