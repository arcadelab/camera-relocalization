import os
import argparse
from pathlib import Path

import general_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Converting COLMAP format to the one we use',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--sequence_root", type=str, required=True, help='root of video sequence')
    args = parser.parse_args()

    sequence_root = Path(args.sequence_root)
    print(str(sequence_root) + ": ")
    result_root = sequence_root / "colmap"

    if not result_root.exists():
        print("ERROR: COLMAP sparse reconstruction does not exist")
    
    result_path_list = list(result_root.glob("*"))
    result_path_list.sort()

    if len(result_path_list) > 1:

        result_visible_views = [len(general_utils.read_visible_view_indexes(prefix)) for prefix in result_path_list]
        best_result = result_visible_views.index(max(result_visible_views))

        if best_result != 0:
            print("Exchanging results in {} for 0".format(best_result))
            # Move best results to 'best' directory
            best_result_path = result_root / str(best_result)
            best_result_dest_path = result_root / "best"

            os.system("cp -r {} {}".format(str(best_result_path), str(best_result_dest_path)))
            os.system("rm -r {}".format(str(best_result_path)))

            # Exchange previous '0' for previous best index
            zero_result_path = result_root / "0"
            os.system("mv {} {}".format(str(zero_result_path), str(best_result_path)))

            # Change 'best' directory to new '0'
            os.system("mv {} {}".format(str(best_result_dest_path), str(zero_result_path)))

        else:
            print("No change needed")

    else:
        print("No change needed")