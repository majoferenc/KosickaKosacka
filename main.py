import argparse
import api_util as api
from model import Model

cli = argparse.ArgumentParser()
cli.add_argument("--maps", nargs="+", default=[],)
cli.add_argument("--base_url", nargs="?", type=str, default="http://localhost/")
cli.add_argument("--render_mode", nargs="?", type=bool, default=False)
args = cli.parse_args()

print("maps: %r" % args.maps)
print("base_url: %s" % args.base_url)
print("render_mode: %s" % str(args.render_mode))

for map in args.maps:
    print("map: %s" % map)
    init_response, response_code = api.init_session(map, args.base_url)
    model = Model(init_response, args.base_url, args.render_mode)
    model.execute()
