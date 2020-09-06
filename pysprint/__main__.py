import sys

from pysprint.templates.build import render, get_parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    sys.exit(render(args.template))