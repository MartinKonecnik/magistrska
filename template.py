# Martin Konečnik, http://git.siwim.si/templates/python-template
# A minimal example for using cestel_helpers and command line parameters. This is meant to be beginner-friendly. If something is unclear, please let me know.
import argparse
import sys
import time
from logging import DEBUG, INFO, WARNING

# Library cestel_helpers is available at https://pip.siwim.si/simple.
# You should add above repository to %APPDATA%/pip/pip.ini under section global. Add either key "index-url" to only use this server (and limit yourself only to officially supported library versions) or extra-index-url to use it on top of default one (sensible when doing a lot of experimentation).
from cestel_helpers.console import configure_all
from cestel_helpers.log import init_logger
from cestel_helpers.version import get_version

if __name__ == '__main__':
    # Version is obtained from git tag and for this to work, there needs to exist at least one Git tag formatted as X.Y.Z. For best practices see Versioning in http://wiki.siwim.si/dokuwiki/lib/exe/fetch.php?media=standards:cestel_standards_en.pdf.
    APP_NAME = f'Template v{get_version()}'

    # I recommend having at least one of these options. Personally, I'm starting to replace keyword --debug with --verbose (and keeping the more flexible options), but that's highly subjective.
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', nargs='?', const=DEBUG, default=WARNING, type=int, help='run application with debug output')  # Uses default when parameter isn't passed and const when it's passed without value.
    parser.add_argument('-v', '--verbose', action='store_true', help='alternative to --debug for simpler applications')
    args = parser.parse_args()

    if args.verbose:
        LOG_LEVEL = DEBUG if args.verbose else WARNING
    else:
        LOG_LEVEL = args.debug

    # Logging to main.log. This should be used for basic application information. This logger will always log at least at INFO level.
    logger_main = init_logger('main', to_console=True, level=LOG_LEVEL if LOG_LEVEL <= INFO else INFO, console_level=LOG_LEVEL if LOG_LEVEL <= INFO else INFO)
    # Logging to app.log. This logger will log to file at level LOG_LEVEL and only print warnings to console.
    logger = init_logger('app', to_console=True, level=LOG_LEVEL, console_level=WARNING if LOG_LEVEL <= WARNING else LOG_LEVEL)

    # Initial output
    logger_main.info(f'{APP_NAME} started.')
    logger_main.info(f'Application parameters: {sys.argv[1:]}')
    logger_main.debug(f'Application parameters received by argparse: {vars(args)}')  # By default, argparse simply takes arguments from sys.argv, but if doing something more advanced it could be useful to see what is actually received.

    # Configures some things that should exist in every console application. Mainly, disables Windows 10 QuickEdit and adds logging for when application is stopped by external factors (e.g. closing console window).
    configure_all(app_name=APP_NAME)

    try:
        logger.info('Application is executing.')  # This will only be visible if application -d is set to INFO or lower.
        logger.debug('This is only written to file.')
        time.sleep(5)
    except SystemExit:  # This is where service should exit when killed by systemctl on Linux. Irrelevant on Windows.
        logger_main.warning('SystemExit raised.')

    # Application stopped output.
    logger_main.info(f'{APP_NAME} stopped.')
