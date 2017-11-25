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
	cout << "Initializing filter with (x,y,theta): (" << x << "," << y << "," << theta << ")" << endl;

	default_random_engine rand_gen; // For sampling the normal distribution

	// Create normal distributions for each measurement
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	num_particles = 750; // TODO: Tweak this

	for(int i = 0; i < num_particles; ++i) {
		// Add a particle
		Particle p;
		p.x = dist_x(rand_gen);
		p.y = dist_y(rand_gen);
		p.theta = dist_theta(rand_gen);
		p.weight = 1.0;

		particles.push_back(p);
		weights.push_back(p.weight);
	}

	// PrintParticles("INITIAL PARTICLES:", false);

	// cout << "Finished Initializing!" << endl;
	is_initialized = true;
}

void ParticleFilter::prediction(double Dt, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.

	cout << "Beginning predition step..." << endl;
	cout << "velocity, yaw_rate: " << velocity << ", " << yaw_rate << endl;
	default_random_engine rand_gen; // For sampling the normal distribution

	// Update each particle
	for(int i = 0; i < particles.size(); ++i) {
		double x = particles[i].x;
		double y = particles[i].y;
		double theta = particles[i].theta;

		// Calculate new values from measurements
		if(abs(yaw_rate) > .00001) { // Don't divide by zero
			x += (velocity / yaw_rate) * (sin(theta + yaw_rate * Dt) - sin(theta));
			y += (velocity / yaw_rate) * (cos(theta) - cos(theta + yaw_rate * Dt));
			theta += yaw_rate * Dt;
		}
		else { // yaw_rate is zero
			x += velocity * cos(theta) * Dt;
			y += velocity * sin(theta) * Dt;
		}

		// Create normal distributions for each measurement
		normal_distribution<double> dist_x(x, std_pos[0]);
		normal_distribution<double> dist_y(y, std_pos[1]);
		normal_distribution<double> dist_theta(theta, std_pos[2]);

		particles[i].x = dist_x(rand_gen);
		particles[i].y = dist_y(rand_gen);
		particles[i].theta = dist_theta(rand_gen);
	}

	// PrintParticles("PREDICTED PARTICLES:", false);

	// cout << "Finished prediction step!" << endl;
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(
	double sensor_range,
	double std_landmark[],
	const vector<LandmarkObs> &observations,
	const Map &map_landmarks
) {
	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory: https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 3.33): http://planning.cs.uiuc.edu/node99.html

	cout << "Updating weights..." << endl;

	vector<double> new_weights;
	for(int p_i = 0; p_i < num_particles; ++p_i) {
		Particle p = particles[p_i];

		// Transform each observation into the map coordinate system (from the vehicle's)
		// Associate each transformed obersation with a map landmark
		vector<LandmarkObs> map_observations;
		vector<Map::single_landmark> chosen_landmarks;
		for(int o_i = 0; o_i < observations.size(); ++o_i) {
			LandmarkObs car_obs = observations[o_i];
			// cout << "Car Obs: " << car_obs.x << ", " << car_obs.y << endl;

			LandmarkObs map_obs;
			map_obs.x = p.x + (car_obs.x * cos(p.theta)) - (car_obs.y * sin(p.theta));
			map_obs.y = p.y + (car_obs.x * sin(p.theta)) + (car_obs.y * cos(p.theta));
			// cout << "Map Obs: " << map_obs.x << ", " << map_obs.y << endl;

			// Associate each transformed map observation to a map landmark
			double min_dist = sensor_range;
			Map::single_landmark chosen_landmark;
			for(int l_i = 0; l_i < map_landmarks.landmark_list.size(); ++l_i) {
				Map::single_landmark landmark = map_landmarks.landmark_list[l_i];

				double distance = dist(map_obs.x, map_obs.y, landmark.x, landmark.y);
				if(l_i == 0) { // Just assign the first, regardless of distance, so we always get an association
					min_dist = distance;
					map_obs.id = landmark.id;
					chosen_landmark = landmark;
				}
				else if(distance < min_dist) {
					min_dist = distance;
					map_obs.id = landmark.id;
					chosen_landmark = landmark;
				}
			}

			// cout << "Chosen Landmark: " << chosen_landmark.x << ", " << chosen_landmark.y << endl;
			map_observations.push_back(map_obs);
			chosen_landmarks.push_back(chosen_landmark);
		}

		// Clear the associations in prep for adding new ones
		p.associations.clear();
		p.sense_x.clear();
		p.sense_y.clear();

		// Calculate (and update) the weight for the particle based on all map_observations (with associated landmark)
		// Weight = product of each measurement's Multivariate-Gaussian probability
		double weight = 1.0;
		double sig_x = std_landmark[0];
		double sig_y = std_landmark[1];
		double c1 = 1.0 / (2.0 * M_PI * sig_x * sig_y);
		double c2 = 2.0 * sig_x * sig_x;
		double c3 = 2.0 * sig_y * sig_y;
		for(int m_i = 0; m_i < map_observations.size(); ++m_i) {
			LandmarkObs obs = map_observations[m_i];
			Map::single_landmark landmark = chosen_landmarks[m_i];

			// Set associations on the particle
			p.associations.push_back(landmark.id);
			p.sense_x.push_back(obs.x);
			p.sense_y.push_back(obs.y);

			// cout << endl;
			// cout << obs.x << ", " << obs.y << "; " << landmark.x << ", " << landmark.y << endl;

			double exponent = (pow(obs.x - landmark.x, 2) / c2) + (pow(obs.y - landmark.y, 2) / c3);
			// cout << "exponent: " << exponent << ", e^(-exponent): " << exp(-exponent) << endl;
			double val = c1 * exp(-exponent);
			// cout << "observation probability is: " << val << endl;
			weight *= val;
		}

		if(weight < DBL_EPSILON) weight = DBL_EPSILON;

		// cout << "New weight is: " << weight << endl;
		new_weights.push_back(weight);
	}

	weights = new_weights;
	// cout << "Finished updating weights!" << endl;
}

void ParticleFilter::resample() {
	vector<Particle> new_particles;
	vector<double> new_weights;

	random_device rd;
	mt19937 gen(rd());
  discrete_distribution<> dist(weights.begin(), weights.end());

  dist(gen);
	for(int i = 0; i < num_particles; ++i) {
		int index = dist(gen);
		new_particles.push_back(particles[index]);
		new_weights.push_back(weights[index]);
	}

	particles = new_particles;
	weights = new_weights;

	NormalizeWeights();

	// PrintParticles("RESAMPLED PARTICLES:", true);
}

// void ParticleFilter::resample() {
// 	// Resample particles with replacement with probability proportional to their weight.
// 	// NOTE: You may find std::discrete_distribution helpful here: http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
// 	cout << "Resampling..." << endl;
//
// 	default_random_engine rand_gen;
// 	uniform_real_distribution<double> distribution(0.0, 1.0);
//
// 	vector<Particle> new_particles;
// 	vector<double> new_weights;
// 	int index = int(num_particles * distribution(rand_gen));
// 	double beta = 0.0;
//
// 	// Collect the max weight
// 	double max_weight = 0.0;
// 	for(int i = 0; i < weights.size(); ++i) {
// 		if(weights[i] > max_weight) max_weight = weights[i];
// 	}
//
// 	// Resample
// 	for(int i = 0; i < num_particles; ++i) {
// 		beta += 2.0 * max_weight * distribution(rand_gen);
// 		while(beta > weights[index]) {
// 			beta -= weights[index];
// 			index = (index + 1) % num_particles;
// 		}
// 		new_particles.push_back(particles[index]);
// 		new_weights.push_back(weights[index]);
// 	}
//
// 	particles = new_particles;
// 	weights = new_weights;
//
// 	NormalizeWeigths();
//
// 	PrintParticles("RESAMPLED PARTICLES:", true);
// }

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

void ParticleFilter::NormalizeWeights() {
	// Normalize weights
	double sum = 0.0;
	for(int i = 0; i < weights.size(); ++i) {
		sum += weights[i];
	}
	// cout << "Sum is: " << sum << endl;
	for(int i = 0; i < weights.size(); ++i) {
		if(sum > 1.0e-50) { // This is preventing a division by zero
			weights[i] /= sum;
			particles[i].weight = weights[i];
		}
		else {
			weights[i] = DBL_EPSILON;
			particles[i].weight = DBL_EPSILON;
		}
	}
}

void ParticleFilter::PrintParticles(string heading, bool show_weights) {
	// Print out particle positions
	cout << heading << endl;
	for(int i = 0; i < particles.size(); ++i) {
		Particle p = particles[i];
		cout << "(x,y): (" << p.x << "," << p.y << ")";
		if(show_weights)
			cout << "   weight: " << p.weight << endl;
		else
			cout << endl;
	}
}
