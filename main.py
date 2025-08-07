# Martin Konečnik, https://git.siwim.si/machine-learning/fix-qa-binary-classification
#
import argparse
import sys
from logging import DEBUG, INFO
from pathlib import Path

import numpy as np
import torch
from cestel_helpers.log import init_logger
from cestel_helpers.version import get_version
from swm import factory

from classifier import BinaryClassifier

if __name__ == '__main__':
    # Version is obtained from git tag. If it does not exist, file .version is used as fallback. If that doesn't exist either, an exception is raised.
    APP_NAME = f'Binary Classification Evaluator v{get_version()}'

    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='model to load')
    parser.add_argument('event', type=str, help='event to evaluate')
    parser.add_argument('-v', '--verbose', action='store_true', help='increase verbosity')
    args = parser.parse_args()

    LOG_LEVEL = DEBUG if args.verbose else INFO

    logger = init_logger('', to_console=True, level=LOG_LEVEL, console_level=LOG_LEVEL)

    # Initial output
    logger.info(f'{APP_NAME} started.')
    logger.info(f'Application parameters: {sys.argv[1:]}')
    logger.debug(f'Application parameters received by argparse: {vars(args)}')

    try:
        args.model = Path(args.model)
        if not args.model.exists():
            logger.error(f'Model {args.model} does not exist')
            sys.exit(1)

        max_length = int(open(args.model / 'dimensions').read())

        model = BinaryClassifier(max_length)
        result = model.load_state_dict(torch.load(args.model / 'fix-qa-binary-classification.pth'))
        logger.debug(f'Loading status: {result}')
        model.eval()

        data = factory.read_file(args.event)
        classification = model(torch.from_numpy(np.array([np.pad(data.acqdata.a[0].data, (0, max_length - len(data.acqdata.a[0].data)))])))
        logger.info(f'Event classified as: {classification.item()}.')  # Closer to 0 means unaltered, 1 means corrected
    except SystemExit:  # This is where service should exit when killed by systemctl on Linux. Irrelevant on Windows.
        logger.warning('SystemExit raised.')

    # Application stopped output.
    logger.info(f'{APP_NAME} stopped.')
