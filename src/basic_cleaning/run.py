#!/usr/bin/env python
"""
Download from W&B the raw dataset, clean it, filter outliers by price and geolocation,
convert last_review to datetime, and upload as a new artifact.
"""
import argparse
import logging
import wandb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):
    """Main function to download, clean, and upload dataset to W&B."""
    
    run = wandb.init(
        project="nyc_airbnb",
        group="cleaning",
        job_type="basic_cleaning",
        save_code=True
    )
    run.config.update(args)

    # Download input artifact
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)

    # Filter price outliers
    idx_price = df['price'].between(args.min_price, args.max_price)
    df = df[idx_price].copy()

    # Filter longitude/latitude for NYC (Step 6)
    idx_geo = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx_geo].copy()

    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])

    # TODO: drop rows outside NYC boundaries
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    
    # Save cleaned data
    cleaned_file = "clean_sample.csv"
    df.to_csv(cleaned_file, index=False)

    # Log cleaned artifact to W&B
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description
    )
    artifact.add_file(cleaned_file)
    run.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic cleaning for NYC Airbnb dataset")
  
    parser.add_argument("--input_artifact", type=str, required=True,
                        help="W&B artifact name for the initial dataset to clean")
    parser.add_argument("--output_artifact", type=str, required=True,
                        help="Name for the cleaned dataset artifact to upload to W&B")
    parser.add_argument("--output_type", type=str, required=True,
                        help="Type of the output artifact (e.g., 'clean_sample')")
    parser.add_argument("--output_description", type=str, required=True,
                        help="Description of the cleaned dataset artifact")
    parser.add_argument("--min_price", type=float, required=True,
                        help="Minimum price of listings to include")
    parser.add_argument("--max_price", type=float, required=True,
                        help="Maximum price of listings to include")

    args = parser.parse_args()
    go(args)
