import logging
import argparse
import json

from controller_thread import ControllerThread


if __name__ == "__main__":

    logger = logging.getLogger(__file__)
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="estimate_age_config.json")

    args = parser.parse_args()

    config_file = args.config_file

    with open(config_file, mode='rt') as fp:

        config = json.load(fp)

    logger.info("Config: {}".format(config))

    controller_thread = ControllerThread(config, logger)
    controller_thread.start()

