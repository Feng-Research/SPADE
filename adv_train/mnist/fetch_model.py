from google_drive_downloader import GoogleDriveDownloader as gdd
import argparse

parser = argparse.ArgumentParser(description='fetch pretrained model')
parser.add_argument('--method', type=str, default="spade",
        help="choose training method, [spade, random, pgd, clean]")


args = parser.parse_args()
clean = "1ooDifu31h8zs-2dr9LkBArbRAFpxret-"
pgd = "1kOIbXGO3FCKMUo0kHD4xA9s0TXfocMsq"
spade = "1javi7s8xnVcmaYQ7rB5TAGJewBsZ-xxp"
random = "1FMSRCaMT9KfCS_4ZYquJRvybQZZLqJxX"

if args.method == "clean":
    fid = clean
    zip_name = "pgd_0.0.zip"
elif args.method == "pgd":
    fid = pgd
    zip_name = "pgd_0.4.zip"
elif args.method == "spade":
    fid = spade
    zip_name = "pgd-spade_0.2_0.4.zip"
elif args.method == "random":
    fid = random
    zip_name = "pgd-random_0.2_0.4.zip"

gdd.download_file_from_google_drive(file_id=fid,
                                            dest_path="./models/{}".format(zip_name),
                                            unzip=True)
