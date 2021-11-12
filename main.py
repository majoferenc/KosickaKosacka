import argparse
import api_util as api

BASE_URL = "http://169.51.194.78:31798/"

cli = argparse.ArgumentParser()
cli.add_argument("--maps", nargs="*", default=[],)
args = cli.parse_args()

print("maps: %r" % args.maps)

for map in args.maps:
    print("map: %s" % map)
    init_response = api.init_session(map)
