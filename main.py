import argparse
import api_util as api
from model import Model

cli = argparse.ArgumentParser()
cli.add_argument("--maps", nargs="+", default=[],)
args = cli.parse_args()

print("maps: %r" % args.maps)

for map in args.maps:
    print("map: %s" % map)
    init_response = api.init_session(map)
    model = Model(init_response)
    model.execute()
