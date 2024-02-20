
import os 
import csv 
import click


@click.command()
@click.option(
    "--input-dir",
    "-i",
    default="data",
    help="The directory containing the input CSV files",
    type=str,
)
@click.option(
    "--outfile",
    "-o",
    default="data/combined_data.csv",
    help="The file path to write the combined data to",
    type=str,
)
@click.option(
    "--prefix",
    "-p",
    default="stocks_data",
    help="The prefix of the input CSV files",
    type=str,
)
def combine_csv_files(input_dir: str, outfile: str, prefix: str) -> None:
    csv_out = open(outfile, "w")
    writer = None
    print(os.listdir(input_dir))
    for file in os.listdir(input_dir):
        if file.startswith(prefix):
            csv_in = open(f"{input_dir}/{file}", "r")
            reader = csv.reader(csv_in)
            if writer is None:
                writer = csv.writer(csv_out)
            next(reader)  # Skip the header
            for row in reader:
                writer.writerow(row)
            csv_in.close()
    csv_out.close()


if __name__ == "__main__":
    combine_csv_files()