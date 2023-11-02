from time import sleep
from tqdm import tqdm
from multiprocessing import Pool


def crunch(numbers):
    # print(numbers)  # commented out to not mess the tqdm output
    sleep(2)


if __name__ == "__main__":
    with Pool(processes=4) as pool:
        progress_bar = tqdm(total=40)
        print("mapping ...")
        results = tqdm(pool.map(crunch, range(40)), total=40)
        print("running ...")
        tuple(results)  # fetch the lazy results
        print("done")