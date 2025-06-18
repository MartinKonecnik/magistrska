# Martin Konečnik, http://git.siwim.si/templates/python-template
# A minimal example for using cestel_helpers and command line parameters. This is meant to be beginner-friendly. If something is unclear, please let me know.
import argparse
import struct
import sys
import time
from logging import DEBUG, INFO, WARNING
from pathlib import Path

import tensorflow as tf
# Library cestel_helpers is available at https://pip.siwim.si/simple.
# You should add above repository to %APPDATA%/pip/pip.ini under section global. Add either key "index-url" to only use this server (and limit yourself only to officially supported library versions) or extra-index-url to use it on top of default one (sensible when doing a lot of experimentation).
from cestel_helpers.console import configure_all
from cestel_helpers.log import init_logger
from cestel_helpers.version import get_version
from lxml import etree
from swm import factory

if __name__ == '__main__':
    print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

    # Version is obtained from git tag and for this to work, there needs to exist at least one Git tag formatted as X.Y.Z. For best practices see Versioning in http://wiki.siwim.si/dokuwiki/lib/exe/fetch.php?media=standards:cestel_standards_en.pdf.
    APP_NAME = f'Binary Classification Trainer v{get_version()}'

    # I recommend having at least one of these options. Personally, I'm starting to replace keyword --debug with --verbose (and keeping the more flexible options), but that's highly subjective.
    parser = argparse.ArgumentParser()
    parser.add_argument('unaltered', help='original swd file')
    parser.add_argument('corrected', help='corrected swd file')
    parser.add_argument('events', help='folder with events')
    parser.add_argument('-v', '--verbose', action='store_true', help='increase verbosity')
    args = parser.parse_args()

    args.events = Path(args.events)

    LOG_LEVEL = DEBUG if args.verbose else INFO

    logger = init_logger('', to_console=True, level=LOG_LEVEL, console_level=LOG_LEVEL)

    # Initial output
    logger.info(f'{APP_NAME} started.')
    logger.info(f'Application parameters: {sys.argv[1:]}')
    logger.debug(f'Application parameters received by argparse: {vars(args)}')  # By default, argparse simply takes arguments from sys.argv, but if doing something more advanced it could be useful to see what is actually received.

    # Configures some things that should exist in every console application. Mainly, disables Windows 10 QuickEdit and adds logging for when application is stopped by external factors (e.g. closing console window).
    configure_all(app_name=APP_NAME)

    try:
        unaltered = etree.parse(args.unaltered).xpath('/swd/site/vehicles/vehicle')
        unaltered_dict = {vehicle.find('wim/ts').text: vehicle.find('wim') for vehicle in unaltered}
        corrected = etree.parse(args.corrected).xpath('/swd/site/vehicles/vehicle')
        corrected_dict = {vehicle.find('wim/ts').text: vehicle.find('wim') for vehicle in corrected}

        logger.info(f'Unaltered: {len(unaltered)}')
        logger.info(f'Corrected: {len(corrected)}')

        # Find matches
        existing_vehicles = [vid for vid in unaltered_dict if vid in corrected_dict.keys()]
        missing_vehicles = [vid for vid in unaltered_dict if vid not in corrected_dict.keys()]

        fixed_vehicles = []

        # TODO Determine what makes a vehicle "fixed"
        for ts in existing_vehicles:
            if unaltered_dict[ts].find('naxles').text != corrected_dict[ts].find('naxles').text:
                fixed_vehicles.append(ts)

        logger.info(f'Existing vehicles: {len(existing_vehicles)}')
        logger.info(f'Fixed vehicles: {len(fixed_vehicles)}')
        logger.info(f'Missing vehicles: {len(missing_vehicles)}')

        signals = []
        binary_labels = []
        for ts in existing_vehicles:
            ets = unaltered_dict[ts].find('ets').text
            path = args.events / f'{ets[:10]}-hh-mm-ss' / f'{ets[:13]}-mm-ss' / f'{ets}.event'
            try:
                data = factory.read_file(path)
                signals.append(data.acqdata.a[16].data)
                binary_labels.append(1 if ts in fixed_vehicles else 0)
            except struct.error:
                pass
    except SystemExit:  # This is where service should exit when killed by systemctl on Linux. Irrelevant on Windows.
        logger.warning('SystemExit raised.')

    # Application stopped output.
    logger.info(f'{APP_NAME} stopped.')
