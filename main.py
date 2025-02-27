import argparse
import sys
from python_repo_template.func import random_sum
import loguru

if __name__ == "__main__":
    logger = loguru.logger
    logger.remove()
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <lvl>{level: <8}</lvl> | {message}",
    )

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--number", type=int, help="Input number")
    args = arg_parser.parse_args()
    number = args.number

    logger.info("-" * 50)
    logger.info(f"Input number: {number}")
    logger.info(f"Output number: {random_sum(number)}")
    logger.info("-" * 50)
