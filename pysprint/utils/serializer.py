import json


def _prepare_json_fragment(ifg, filename, x_before_transform, y_before_transform, verbosity=0):
    if hasattr(ifg, "phase") and ifg.phase is not None:
        phase_or_GD = ifg.phase
    elif hasattr(ifg, "GD") and ifg.GD is not None:
        phase_or_GD = ifg.GD
    else:
        phase_or_GD = False
    if verbosity > 0:
        return {
            filename: {
                "method": str(ifg.__class__).split(".")[-1].strip("''>"),
                "GD": getattr(phase_or_GD, "GD", 0),
                "GDD": getattr(phase_or_GD, "GDD", 0),
                "TOD": getattr(phase_or_GD, "TOD", 0),
                "FOD": getattr(phase_or_GD, "FOD", 0),
                "QOD": getattr(phase_or_GD, "QOD", 0),
                "SOD": getattr(phase_or_GD, "SOD", 0),
                "meta": {} if len(ifg.meta) == 0 else ifg.meta,
                "x": x_before_transform.tolist(),
                "y": y_before_transform.tolist(),
                "phase": {
                    "is_GD": getattr(phase_or_GD, "GD_mode", False),
                    "x": list(getattr(phase_or_GD, "x", [])),
                    "y": list(getattr(phase_or_GD, "y", [])),
                },
            }
        }
    else:
        return {
            filename: {
                "method": str(ifg.__class__).split(".")[-1].strip("''>"),
                "GD": getattr(phase_or_GD, "GD", 0),
                "GDD": getattr(phase_or_GD, "GDD", 0),
                "TOD": getattr(phase_or_GD, "TOD", 0),
                "FOD": getattr(phase_or_GD, "FOD", 0),
                "QOD": getattr(phase_or_GD, "QOD", 0),
                "SOD": getattr(phase_or_GD, "SOD", 0),
                "meta": {} if len(ifg.meta) == 0 else ifg.meta,
            }
        }


def _write_or_update_json_fragment(filename, new_ifg, new_ifg_name):
    # open in read mode, check what operation to run
    with open(filename, "r") as json_file:
        data = json.load(json_file)

    data[new_ifg_name] = new_ifg[new_ifg_name]

    # reopen in write mode (the cursor is at the end of the file
    # after read)
    with open(filename, "w") as json_file:
        json_file.write(json.dumps(data, indent=2))
