import numpy as np
import matplotlib.pyplot as plt
import sys
from configparser import ConfigParser
from pickle import load


def read_file(path_to_file):
    file = open(path_to_file, 'rb')
    return load(file)


def write_lines_to_file(path_to_file, lines):
    file = open(path_to_file, 'w')
    return file.writelines(lines)


def plot_neighbourhoods(
        centre_coordinates,
        neighbourhoods,
        labels,
        title=None,
        point_labels=None):
    markers = ['*', 's', 'x', 'v', '^']
    _ = plt.figure()
    axis = plt.axes(projection='3d')
    axis.scatter3D(
        centre_coordinates[0],
        centre_coordinates[1],
        centre_coordinates[2], marker='o', label='centre')
    for index, label in enumerate(labels):
        axis.scatter3D(
            neighbourhoods[index].T[0],
            neighbourhoods[index].T[1],
            neighbourhoods[index].T[2], marker=markers[index], label=label)
    if point_labels:
        for txt in point_labels.keys():
            axis.text(
                point_labels[txt][0],
                point_labels[txt][1],
                point_labels[txt][2],
                '%s' % (str(txt)), size=20, zorder=1,
                color='k')
    plt.title(title)
    plt.legend()
    plt.show()


def print_and_plot(
        results,
        num_of_rots,
        reference_positions,
        site):

    centre_coordinates = np.array(results['centre_atom'][site])
    current_neighbourhood = np.array(results['neighbourhood'][site])
    orig_neighbourhood = np.array(results['neighbourhood_orig'][site])
    reshuffled_pattern = np.array(results['reshuffled_pattern'][site])
    rotated_pattern = np.array(results['rotated_pattern'][site])
    optimal_pattern = np.array(results['pattern'][site])

    # plot neighbourhoods
    plot_neighbourhoods(
        centre_coordinates,
        [
            orig_neighbourhood,
            current_neighbourhood
            ],
        labels=['original neighb.', 'current neighb.'],
        title='neighbourhood of {0}'.format(centre_coordinates)
        )

    # plot references
    plot_neighbourhoods(
        centre_coordinates,
        [
            np.array(reference_positions),
            current_neighbourhood,
            rotated_pattern
        ],
        labels=['original ref.', 'current neighb.', 'rot ref.'],
        title='rescaled and rotated reference')

    point_labels = {}
    for ind in results['random_indices'][site]:
        point_labels.update(
            {'n{0}'.format(ind): current_neighbourhood[int(ind)]})
        point_labels.update(
            {'r{0}'.format(ind): reshuffled_pattern[int(ind)]})
    print('mapping after reshuffling')
    print('for the last rotation (n–neighbour site, r–reference):')
    print(point_labels)

    plot_neighbourhoods(
        centre_coordinates,
        [rotated_pattern, current_neighbourhood],
        labels=['rotated ref.', 'current neighb.'],
        title='rotated ref. and ' +
        'labels for indices {0}'.format(results['random_indices'][site]),
        point_labels=point_labels
        )
    print('score for atom {0}:'.format(site))
    print(results['score'][site])
    print('Mean score for the system:')
    print(np.mean(results['score']))

    plot_neighbourhoods(
        centre_coordinates,
        [
            optimal_pattern,
            current_neighbourhood],
        labels=['optimal ref.', 'current neighb.'],
        title='final reference with score {0}'.format(
            results['score'][site]
            ))


if __name__ == '__main__':
    # Collect information
    parser = ConfigParser()
    parser.read('params.ini')
    reference_positions = np.loadtxt(
            parser['USER']['file_with_reference'])
    num_of_rots = parser.getint('USER', 'number_of_rotations')

    site = int(sys.argv[1])

    # Provide the PyGem log object
    results = read_file(sys.argv[2])

    print_and_plot(
        results,
        num_of_rots,
        reference_positions,
        site)
