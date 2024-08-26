from utils import *
import sys
import os
import warnings

print("\n\n-- Libraries Imported")

if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    if not os.path.isfile(sys.argv[1]):
        print("File doesn't exist")
        raise SystemExit(1)

    file = open(sys.argv[1], encoding="utf-8")
    text = file.read()

    prjName = sys.argv[2]

    text, global_ents_list = resolveText(text)

    generateGraph(prune_self_loops(prune_infreq_objects(prune_infreq_subjects(extractTriplets(text, global_ents_list, prjName=prjName)))), prjName=prjName)
