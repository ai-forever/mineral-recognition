import pandas as pd
import argparse

from scripts.utils import configure_logging

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error


def main(args):

    logger = configure_logging()

    data_size = pd.read_csv(args.predict_size)
    size_gt_agaf = pd.read_csv(args.ground_true_size) # DEBUG

    sizes = data_size.merge(size_gt_agaf, left_on='image_path', right_on='path_final')

    mae = mean_absolute_error(sizes['SIZESM'], sizes['contour_analysis_mineral_x_in_cm'])
    r2 = r2_score(sizes['SIZESM'], sizes['contour_analysis_mineral_x_in_cm'])
    mse = mean_squared_error(sizes['SIZESM'], sizes['contour_analysis_mineral_x_in_cm'], squared=False)
    mape = mean_absolute_percentage_error(sizes['SIZESM'], sizes['contour_analysis_mineral_x_in_cm'])

    logger.info(f'Mean absolute error: {mae:.4f}')
    logger.info(f'R2 score: {r2:.4f}')
    logger.info(f'Mean squared error: {mse:.4f}')
    logger.info(f'Mean absolute percentage error: {mape:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ground_true_size', type=str,
                        default='razmery_merged_agaf.csv',
                        help='Path to csv.file with initial sizes of minerals.')
    parser.add_argument('--predict_size', type=str,
                        default='size_centimeter_minerals.csv',
                        help='Path to csv.file with predict sizes of minerals.')

    args = parser.parse_args()

    main(args)
