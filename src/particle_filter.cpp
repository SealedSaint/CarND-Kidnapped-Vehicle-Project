/*
* particle_filter.cpp
*
*  Created on: Dec 12, 2016
*      Author: Tiffany Huang
*/

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	default_random_engine rand_gen; // For sampling the normal distribution

	// Create normal distributions for each measurement
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	num_particles = 1000; // TODO: Tweak this
	for(int i = 0; i < num_particles; ++i) {
		// Add a particle
		Particle p;
		p.x = dist_x(rand_gen);
		p.y = dist_y(rand_gen);
		p.theta = dist_theta(rand_gen);
		p.weight = 1.0;

		particles.push_back(p);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double Dt, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.

	default_random_engine rand_gen; // For sampling the normal distribution

	// Update each particle
	for(int i = 0; i < particles.size(); ++i) {
		double x = particles[i].x;
		double y = particles[i].y;
		double theta = particles[i].theta;

		// Calculate new values from measurements
		// TODO: Understand why x is determined by sin and y is determined by cos
		if(abs(yaw_rate) < .001) { // Don't divide by zero
			x += (velocity / yaw_rate) * (sin(theta + yaw_rate * Dt) - sin(theta));
			y += (velocity / yaw_rate) * (cos(theta) - cos(theta + yaw_rate * Dt));
			theta += yaw_rate * Dt;
		}
		else { // yaw_rate is zero
			x += velocity * sin(theta) * Dt;
			y += velocity * cos(theta) * Dt;
		}

		// Create normal distributions for each measurement
		normal_distribution<double> dist_x(x, std_pos[0]);
		normal_distribution<double> dist_y(y, std_pos[1]);
		normal_distribution<double> dist_theta(theta, std_pos[2]);

		particles[i].x = dist_x(rand_gen);
		particles[i].y = dist_y(rand_gen);
		particles[i].theta = dist_theta(rand_gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(
	double sensor_range, double std_landmark[],
	const std::vector<LandmarkObs> &observations, const Map &map_landmarks
) {
		// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
		//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
		// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
		//   according to the MAP'S coordinate system. You will need to transform between the two systems.
		//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
		//   The following is a good resource for the theory: https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
		//   and the following is a good resource for the actual equation to implement (look at equation 3.33): http://planning.cs.uiuc.edu/node99.html

		for(int p_i = 0; p_i < num_particles; ++p_i) {
			Particle p = particles[p_i];

			// Transform each observation into the map coordinate system (from the vehicle's)
			// Associate each transformed obersation with a map landmark
			vector<LandmarkObs> map_observations;
			for(int o_i = 0; o_i < observations.size(); ++o_i) {
				LandmarkObs car_obs = observations[o_i];

				LandmarkObs = map_obs;
				map_obs.x = p.x + (car_obs.x * cos(p.theta)) - (car_obs.y * sin(p.theta));
				map_obs.y = p.y + (car_obs.x * sin(p.theta)) + (car_obs.y * cos(p.theta));

				// Associate each transformed map observation to a map landmark
				double min_dist = sensor_range;
				for(int l_i = 0; i < map_landmarks.landmark_list.size(); ++l_i) {
					single_landmark landmark = map_landmarks.landmark_list[l_i];

					double distance = dist(map_obs.x, landmark.x, map_obs.y, landmark.y);
					if(l_i == 0) { // Just assign the first, regardless of distance, so we always get an association
						min_dist = distance
						map_obs.id = landmark.id;
					}
					else if(distance < min_dist) {
						min_dist = distance;
						map_obs.id = landmark.id;
					}
				}

				map_observations.push_back(obs);
			}

			// Calculate (and update) the weight for the particle based on all map_observations (with associated landmark)
			// Weight = product of each measurement's Multivariate-Gaussian probability
			double weight = 1;
			double sig_x = std_landmark[0];
			double sig_y = std_landmark[1];
			double c1 = 1 / (2 * M_PI * sig_x * sig_y);
			for(int m_i = 0; m_i < map_observations.size(); ++m_i) {
				LandmarkObs obs = map_observations[m_i];

				// Get the associate landmark
				single_landmark landmark;
				for(int l_i = 0; l_i < map_landmarks.landmark_list.size(); ++l_i) {
					single_landmark l = map_landmarks.landmark_list[l_i];

					if(l.id == obs.id) {
						landmark = l;
						break;
					}
				}

				double exponent = ((obs.x - landmark.x)**2 / (2 * sig_x**2) + (obs.y - landmark.y)**2 / (2 * sig_y**2));

				weight *= c1 * exp(-exponent);
			}

			p.weight = weight;
		}
	}

	void ParticleFilter::resample() {
		// Resample particles with replacement with probability proportional to their weight.
		// NOTE: You may find std::discrete_distribution helpful here: http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

		default_random_engine rand_gen;
		uniform_real_distribution<double> distribution(0.0, 1,0);

		vector<Particle> new_particles;
		int index = int(num_particles * distribution(rand_gen));
		double beta = 0;

		double max_weight = 0.0;
		for(int i = 0; i < weights.size(); ++i) {
			if(weights[i] > max_weight) max_weight = weights[i];
		}

		for(int i = 0; i < num_particles; ++i) {
			beta += 2 * max_weight * distribution(rand_gen);
			while(beta > weights[index]) {
				beta -= weights[index];
				index = (index + 1) % num_particles;
			}
			new_particles.push_back(particles[index])
		}

		particles = new_particles;
	}

	Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y) {
		// particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
		// associations: The landmark id that goes along with each listed association
		// sense_x: the associations x mapping already converted to world coordinates
		// sense_y: the associations y mapping already converted to world coordinates

		//Clear the previous associations
		particle.associations.clear();
		particle.sense_x.clear();
		particle.sense_y.clear();

		particle.associations= associations;
		particle.sense_x = sense_x;
		particle.sense_y = sense_y;

		return particle;
	}

	string ParticleFilter::getAssociations(Particle best) {
		vector<int> v = best.associations;
		stringstream ss;
		copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
		string s = ss.str();
		s = s.substr(0, s.length()-1);  // get rid of the trailing space
		return s;
	}

	string ParticleFilter::getSenseX(Particle best) {
		vector<double> v = best.sense_x;
		stringstream ss;
		copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
		string s = ss.str();
		s = s.substr(0, s.length()-1);  // get rid of the trailing space
		return s;
	}

	string ParticleFilter::getSenseY(Particle best) {
		vector<double> v = best.sense_y;
		stringstream ss;
		copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
		string s = ss.str();
		s = s.substr(0, s.length()-1);  // get rid of the trailing space
		return s;
	}
