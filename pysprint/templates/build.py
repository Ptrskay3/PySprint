import sys
import argparse
import pathlib
import datetime
import jinja2


NAME_MAPPING = {
    "cff": 'CosFitMethod',
    "mm": 'MinMaxMethod',
    "spp": 'SPPMethod',
    "fft": "FFTMethod",
    "wft": "WFTMethod"
}


def render(methodname):
    templateEnv = jinja2.Environment(loader=jinja2.PackageLoader('pysprint', 'templates'))
    TEMPLATE_FILE = "method_template.py_t"
    template = templateEnv.get_template(TEMPLATE_FILE)
    BODY_FILE = methodname + "_body.py_t"
    body = templateEnv.get_template(BODY_FILE)
    t = template.render(
        methodname=NAME_MAPPING[methodname],
        body=body.render(),
        date=datetime.datetime.now()
    )
    if methodname == "spp":
        SPP_TEMPLATE = "spp_body.py_t"
        template = templateEnv.get_template(SPP_TEMPLATE)
        t = template.render(
            methodname=NAME_MAPPING[methodname],
            date=datetime.datetime.now()
        )
    path = pathlib.Path(".")
    filename = methodname + "-" + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M') + ".py"
    with open(path / filename, "w") as output_file:
        output_file.write(t)
    print(f"Created template {filename} at {path.absolute()}")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', '--template', choices=NAME_MAPPING.keys())
    return parser


def main(argv=sys.argv[1:]):
    parser = get_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as err:
        return err.code
    render(args.template)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
