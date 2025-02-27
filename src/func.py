import random


def random_sum(num: int) -> int:
    """
    Add a random number between 0 and 100 to the input number.
    """
    return num + random.randint(0, 100)
