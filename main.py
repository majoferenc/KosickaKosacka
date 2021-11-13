import argparse
import api_util as api
from model import Model

cli = argparse.ArgumentParser()
cli.add_argument("--maps", nargs="+", default=[],)
cli.add_argument("--base_url", nargs="+", default="http://localhost/")
args = cli.parse_args()

print("maps: %r" % args.maps)
print("base_url: %s" % str(args.base_url))

for map in args.maps:
    print("map: %s" % map)
    init_response = api.init_session(map, args.base_url[0])
    model = Model(init_response, args.base_url[0])
    model.execute()
