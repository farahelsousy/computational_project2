
from exercise5 import exercise5
from exercise6 import exercise6
from exercise7 import exercise7
from exercise8 import exercise8
from exercise9 import exercise9
import plot_results
import farms_pylog as pylog


def main():

    pylog.info("Running Project 2 exercises")

    exercise5()
    exercise6()
    exercise7()
    exercise8()
    exercise9()
    plot_results.main()


if __name__ == '__main__':
    main()

