import main_bert
import argparse

parser = argparse.ArgumentParser(description='Makes Predition for Punctuation Marks')
parser.add_argument('--list', type=str, nargs='+', default=[])

args = parser.parse_args()

if args.list == []:
    print("Enter one arg at least")

else:
    print(args.list)
    for ar in args.list:
        print(f"Running the {ar} config")
        main_bert.main(ar)
        print(f" ** {ar} config COMPLETE!")

