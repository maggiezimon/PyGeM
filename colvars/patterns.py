import jax
from jax import numpy as np
from jaxopt import GradientDescent as minimize
from .core import PatternCV
from pysages.utils import gaussian
import time


# Global constants taken from https://github.com/cpgoodri/jax_transformations3d
# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0),
    'sxyx': (0, 0, 1, 0),
    'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0),
    'syzx': (1, 0, 0, 0),
    'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0),
    'syxy': (1, 1, 1, 0),
    'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0),
    'szyx': (2, 1, 0, 0),
    'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1),
    'rxyx': (0, 0, 1, 1),
    'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1),
    'rxzy': (1, 0, 0, 1),
    'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1),
    'ryxy': (1, 1, 1, 1),
    'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1),
    'rxyz': (2, 1, 0, 1),
    'rzyz': (2, 1, 1, 1)
}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())


jax.config.update("jax_enable_x64", True)
# def transpose(x: np.ndarray) -> np.ndarray:
#     return np.array([np.stack(t) for t in zip(*x)])


def quaternion_from_euler(ai, aj, ak, axes='sxyz'):
    """Return quaternion from Euler angles and axis sequence.

    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple

    Code is taken from https://github.com/cpgoodri/jax_transformations3d
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # noqa: validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis + 1
    j = _NEXT_AXIS[i + parity - 1] + 1
    k = _NEXT_AXIS[i - parity] + 1

    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj

    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = np.cos(ai)
    si = np.sin(ai)
    cj = np.cos(aj)
    sj = np.sin(aj)
    ck = np.cos(ak)
    sk = np.sin(ak)
    cc = ci * ck
    cs = ci * sk
    sc = si * ck
    ss = si * sk

    q = np.empty((4,))
    if repetition:
        q = q.at[0].set(cj * (cc - ss))
        q = q.at[i].set(cj * (cs + sc))
        q = q.at[j].set(sj * (cc + ss))
        q = q.at[k].set(sj * (cs - sc))
    else:
        q = q.at[0].set(cj * cc + sj * ss)
        q = q.at[i].set(cj * sc - sj * cs)
        q = q.at[j].set(cj * ss + sj * cc)
        q = q.at[k].set(cj * cs - sj * sc)
    if parity:
        q = q.at[j].multiply(-1.0)
    return q


def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.
    Code is taken from https://github.com/cpgoodri/jax_transformations3d
    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)

    def calc_mat_posn(qn):
        q, n = qn
        q *= np.sqrt(2.0 / n)
        q = np.outer(q, q)
        return np.array(
          [
              [
                  1.0 - q[2, 2] - q[3, 3],
                  q[1, 2] - q[3, 0],
                  q[1, 3] + q[2, 0],
                  0.0
                ],
              [
                  q[1, 2] + q[3, 0],
                  1.0 - q[1, 1] - q[3, 3],
                  q[2, 3] - q[1, 0],
                  0.0
                ],
              [
                  q[1, 3] - q[2, 0],
                  q[2, 3] + q[1, 0],
                  1.0 - q[1, 1] - q[2, 2],
                  0.0
               ],
              [0.0, 0.0, 0.0, 1.0]])

    return jax.lax.cond(
          n < _EPS, np.identity(4), lambda x: x, (q, n), calc_mat_posn)


def rotate_pattern_with_quaternions(rot_q, pattern):
    return np.transpose(
            np.dot(
                quaternion_matrix(rot_q)[:3, :3],
                np.transpose(pattern)))


def func_to_optimise(Q, modified_pattern, local_pattern):
    return np.linalg.norm(
            rotate_pattern_with_quaternions(
                Q, modified_pattern) -
            local_pattern)

# Main class implementing the GeM CV


class Pattern():
    def __init__(
        self, simulation_box, positions,
        reference, min_cutoff,
        characteristic_distance,
        centre_j_id,
        standard_deviation,
        mesh_size
                ):

        self.positions = positions.copy()
        self.min_cutoff = min_cutoff
        self.characteristic_distance = characteristic_distance
        self.reference = reference
        self.simulation_box = simulation_box
        self.centre_j_id = centre_j_id
        self.centre_j_coords = positions[centre_j_id]
        self.standard_deviation = standard_deviation
        self.mesh_size = mesh_size

    def comp_pair_distance_squared(self, pos1):
        mic_vector = self.positions[self.centre_j_id] - pos1
        periodic_box_vectors = self.simulation_box

        mic_vector -= periodic_box_vectors[2]*np.round(
                mic_vector[2]/periodic_box_vectors[2][2])
        mic_vector -= periodic_box_vectors[1]*np.round(
                mic_vector[1]/periodic_box_vectors[1][1])
        mic_vector -= periodic_box_vectors[0]*np.round(
                mic_vector[0]/periodic_box_vectors[0][0])

        mic_norm = np.linalg.norm(mic_vector)
        # Check if the site is within a given distance;
        # if not, set the distance to a high value to ignore it
        return None, (mic_norm, mic_vector)

    def _generate_neighbourhood(self):
        self._neighbourhood = []

        # Don't perform the calculation for the centre atom j
        _, (distances, mic_vectors) = jax.lax.scan(
                lambda _, pos:
                jax.lax.cond(
                    np.allclose(pos, self.positions[self.centre_j_id]),
                    lambda pos: (None, (1e5, np.array([0.0, 0.0, 0.0]))),
                    lambda pos: self.comp_pair_distance_squared(pos),
                    pos),
                None,
                self.positions)

        _, distances = jax.lax.scan(
                lambda _, d: jax.lax.cond(
                    d > self.min_cutoff,
                    lambda d: (None, d),
                    lambda d: (None, 1e5), d), None, distances)
        ids_of_neighbours = np.argsort(distances)[:len(self.reference)]

        coordinates = mic_vectors[ids_of_neighbours] + self.centre_j_coords
        # Step 1: Translate to origin;
        coordinates = coordinates.at[:].set(
                coordinates-np.mean(coordinates, axis=0))
        for vec_id, mic_vector in enumerate(mic_vectors[ids_of_neighbours]):
            neighbour = {
                         'id': ids_of_neighbours[vec_id],
                         'coordinates': coordinates[vec_id],
                         'mic_vector': mic_vector,
                         'pos_wrt_j': self.centre_j_coords - mic_vector,
                         'distance': distances[ids_of_neighbours[vec_id]]
                         }
            self._neighbourhood.append(neighbour)

        self._neighbour_coords = np.array(
                [n['coordinates'] for n in self._neighbourhood])
        self._orig_neighbour_coords = np.array(
                self.positions[ids_of_neighbours])

    def compute_score(self, optim_reference):
        r = self._neighbour_coords - optim_reference
        return np.prod(gaussian(1, self.standard_deviation, r))
        # return jax.lax.fori_loop(
        #         0, len(self._neighbourhood),
        #         lambda i, x: x*np.exp(
        #             -0.5*(
        #                     (
        #                         self._neighbour_coords[i] -
        #                         optim_reference[i]
        #                     ).dot(
        #                         self._neighbour_coords[i] -
        #                         optim_reference[i]
        #                         ) /
        #                     np.power(self.standard_deviation, 2)
        #                 )),
        #         1.0)

    def rotate_reference(self, random_euler_point):
        # Perform rotation of the reference pattern;
        # Using Euler angles in radians construct a quaternion base;
        # Convert the quaternion to a 3x3 rotation matrix.
        rot_q = quaternion_from_euler(*random_euler_point)
        return rotate_pattern_with_quaternions(rot_q, self.reference)

    def resort(self, pattern_to_resort, key):
        # This subroutine shuffles randomly the input local pattern
        # and resorts the reference indices in order to "minimise"
        # the distance of the corresponding sites

        random_indices = jax.random.permutation(
                key, np.arange(len(self._neighbourhood)),
                axis=0, independent=False)
        random_neighbour_coords = self._neighbour_coords[random_indices]

        def find_closest(carry, neighbour_coords):
            ref_positions = carry
            distances = [
                    np.linalg.norm(
                        vector - neighbour_coords) for vector in ref_positions]
            min_index = np.argmin(np.array(distances))
            positions = ref_positions.at[
                    min_index].set(np.array([-1e10, -1e10, -1e10]))
            new_ref_positions = ref_positions[min_index]
            return positions, new_ref_positions

        _, closest_reference = jax.lax.scan(
                find_closest, pattern_to_resort, random_neighbour_coords)
        # Reorder the reference to match the positions of the neighbours
        reshuffled_reference = np.zeros_like(closest_reference)
        reshuffled_reference = reshuffled_reference.at[
                random_indices].set(closest_reference)
        return reshuffled_reference, random_indices

    def check_settled_status(self, modified_reference):

        def mark_close_sites(_, reference_pos):

            def return_close(_, n):
                return jax.lax.cond(
                        np.linalg.norm(
                            n - reference_pos) <
                        self.characteristic_distance/2.0,
                        lambda x: (None, x+1), lambda x: (None, x), 0)

            _, close_sites_per_reference = jax.lax.scan(
                    return_close, None, self._neighbour_coords)
            return None, close_sites_per_reference

        _, close_sites = jax.lax.scan(
                mark_close_sites, None, modified_reference)
        _, indices = jax.lax.scan(
                lambda _, sites: (
                    None, jax.lax.cond(
                        np.sum(sites) == 1,
                        lambda s: s,
                        lambda s: np.zeros_like(s), sites)
                    ),
                None, close_sites)
        # Return the locations of settled nighbours in the neighbourhood;
        # Settlled site should have a unique neighbour
        settled_neighbour_indices = np.where(
                np.sum(indices, axis=0) >= 1, 1, 0)
        return settled_neighbour_indices

    def driver_match(self, number_of_rotations, number_of_opt_steps, num):

        self._generate_neighbourhood()

        '''Step2: Scale the reference so that the spread matches
        with the current local pattern'''
        local_distance = 0.0
        reference_distance = 0.0
        for n_index, neighbour in enumerate(self._neighbourhood):
            local_distance += np.dot(
                    neighbour['coordinates'], neighbour['coordinates'])
            reference_distance += np.dot(
                    self.reference[n_index], self.reference[n_index])

        self.reference *= np.sqrt(local_distance/reference_distance)

        '''Step3: mesh-loop -> Define angles in reduced Euler domain,
        and for each rotate, resort and score the pattern

        The implementation below follows the article Martelli et al. 2018


        (a) Randomly with uniform probability pick a point in the Euler domain,
        (b) Rotate the reference
        (c) Resort the local pattern and assign the closest reference sites,
        (d) Perform the optimisation step (conjugate gradient),
        and (e) store the score with (f) the final settled status'''
        def get_all_scores(newkey, euler_point):
            # b. Rotate the reference pattern
            rotated_reference = self.rotate_reference(euler_point)
            # c. Resort; shuffle the local pattern
            # and assign ids to the closest reference sites
            newkey, newsubkey = jax.random.split(jax.random.PRNGKey(newkey))
            reshuffled_reference, random_indices = self.resort(
                    rotated_reference, newsubkey)
            # d. Find the best rotation that aligns the settled sites
            # in both patterns;
            # Here, ‘optimal’ or ‘best’ is in terms of least squares errors
            solver = minimize(
                    fun=func_to_optimise,
                    maxiter=number_of_opt_steps)
            optim = solver.run(
                    init_params=np.array([0.1, 0.0, 0.0, 0.1]),
                    modified_pattern=reshuffled_reference,
                    local_pattern=self._neighbour_coords
                    )
            optimal_reference = rotate_pattern_with_quaternions(
                    optim.params, reshuffled_reference)
            # e. Compute and store the score
            score = self.compute_score(optimal_reference)
            result = dict(
                        score=score,
                        rotated_pattern=rotated_reference,
                        random_indices=random_indices,
                        reshuffled_pattern=reshuffled_reference,
                        pattern=optimal_reference,
                        quaternions=optim.params
                        )
            return result

        # a. Randomly pick a point in the Euler domain

        key, subkey = jax.random.split(jax.random.PRNGKey(num))
        mesh_size = self.mesh_size
        grid_dimension = 0.25*np.pi/mesh_size
        euler_angles = np.arange(
            0,
            0.125*np.pi+(mesh_size/2+1)*grid_dimension,
            grid_dimension)
        random_points = jax.random.randint(
                subkey, (number_of_rotations, 3), minval=0.0, maxval=mesh_size)
        # Excute find_max_score for each angle
        # and store the result with the highest score

        scoring_results = jax.vmap(
             get_all_scores
             )(num+np.arange(number_of_rotations), euler_angles[random_points])
        optimal_case = np.argmax(scoring_results['score'])

        # f. Check how many settled sites there are
        settled_neighbour_ids = self.check_settled_status(
                    scoring_results[
                        'pattern'][optimal_case])

        # Storing all the data is only for validation and analysis;
        # For FFS, only score willl be returned, i.e., optimal_result['score']
        optimal_result = dict(
                score=scoring_results['score'][optimal_case],
                rotated_pattern=scoring_results[
                    'rotated_pattern'][optimal_case],
                random_indices=scoring_results['random_indices'][optimal_case],
                reshuffled_pattern=scoring_results[
                    'reshuffled_pattern'][optimal_case],
                pattern=scoring_results['pattern'][optimal_case],
                quaternions=scoring_results['quaternions'][optimal_case],
                settled=settled_neighbour_ids,
                centre_atom=self.centre_j_coords,
                neighbourhood=self._neighbour_coords,
                neighbourhood_orig=self._orig_neighbour_coords)
        return optimal_result


def calculate_lom(
    positions: np.array,
    reference_positions: np.array,
        **kwargs):

    box = np.asarray(kwargs['box'])
    number_of_rotations = kwargs['number_of_rotations']
    number_of_opt_it = kwargs['number_of_opt_it']
    min_cutoff = kwargs['min_cutoff']
    characteristic_distance = kwargs['characteristic_distance']
    standard_deviation = kwargs['standard_deviation']
    mesh_size = kwargs['mesh_size']

    ''''Step1: Move the reference and
    local patterns so that their centers coincide with the origin'''

    reference_positions = reference_positions.at[:].set(
            reference_positions-np.mean(reference_positions, axis=0))

    # Calculate scores
    seed = np.int32(time.process_time()*1e5)
    optimal_results = jax.vmap(
                    lambda i: Pattern(
                                      box, positions,
                                      reference_positions,
                                      min_cutoff, characteristic_distance,
                                      i, standard_deviation,
                                      mesh_size).driver_match(
                                          number_of_rotations,
                                          number_of_opt_it,
                                          seed + i*number_of_rotations)
                                      )(
                                    np.arange(
                                            len(positions), dtype=np.int32)
                                        )
    average_score = np.sum(optimal_results['score']) / len(positions)
    # return average_score #  , optimal_results
    return average_score


class GeM(PatternCV):
    @property
    def function(self):
        return lambda rs: calculate_lom(
            rs, np.asarray(self.reference_positions),
            **self.kwargs)
