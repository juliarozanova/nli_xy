import os
import pkgutil


if __name__ == "__main__":
    realpath = "/".join(os.path.realpath(__file__).split("/")[:-1])

    for finder, modname, ispkg in pkgutil.iter_modules([realpath]):
        if (not modname.startswith("__")):
            module = finder.find_module(modname).load_module(modname)
            module.run.main()
