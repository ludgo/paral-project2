#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <string>
#include <vector>
#include <set>
#include <algorithm>

#include <fstream>
#include <iostream>
#include <sstream>

#include <omp.h>

struct comparator_pair_second {
    bool operator ()(const std::pair<unsigned int, float> &a, const std::pair<unsigned int, float> &b) {
        return a.second < b.second;
    }
};

float calculate_distance(const std::vector<float>& a, const std::vector<float>& b) {
    float distance = 0.f;
    for (int i = 0; i < a.size(); i++) {
        float diff = a[i] - b[i];
        distance += (diff * diff);
    }
    return distance;
}

float calculate_jaccard(const std::set<int>& a, const std::set<int>& b) {

    std::set<int> common_neighbors;
    std::set_intersection(a.begin(),a.end(), b.begin(), b.end(),
                          std::inserter(common_neighbors, common_neighbors.begin()));

    std::set<int> all_neighbors;
    std::set_union(a.begin(), a.end(), b.begin(), b.end(),
                          std::inserter(all_neighbors, all_neighbors.begin()));

    return 1.0f * common_neighbors.size() / all_neighbors.size();
}

int main(int argc, char **argv) {

    std::string file_path;
    int k;
    int n_threads = omp_get_max_threads();
    int chunk_size = 1;

    file_path = argv[1];
    k = atoi(argv[2]);
    if (argc > 3 && atoi(argv[3]) != 0) n_threads = atoi(argv[3]);
    if (argc > 4) chunk_size = atoi(argv[4]);

    std::ifstream input_stream(file_path.c_str());
    if (!input_stream.is_open()) {
        std::cout << "Problems opening input file.\n";
        return EXIT_FAILURE;
    }

    std::vector<std::vector<float>> points;
    std::string line;
    while (std::getline(input_stream, line)) {
        std::vector<float> point;
        std::stringstream line_stream(line);
        std::string dimension;
        while (std::getline(line_stream, dimension, '\t')) {
            point.push_back(atof(dimension.c_str()));
        }
        points.push_back(std::move(point));
    }

    std::vector<std::set<int>> knn;
    knn.resize(points.size());

    double start_knn = omp_get_wtime();

#pragma omp parallel for num_threads(n_threads) schedule(dynamic, chunk_size)
    for (int i = 0; i < points.size(); i++) {

        std::set<std::pair<int, float>, comparator_pair_second> knn_i;

        for (int j = 0; j < points.size(); j++) {
            if (i == j) continue;

            float distance = calculate_distance(points[i], points[j]);

            knn_i.insert(std::make_pair(j, distance));

            if (knn_i.size() > k) {
                knn_i.erase(--knn_i.end());
            }
        }

        for (auto it = knn_i.begin(); it != knn_i.end(); it++) {
            knn[i].insert((*it).first);
        }
    }

    double finish_knn = omp_get_wtime();
    printf("kNN in %.8f seconds\n", finish_knn - start_knn);
    //std::ofstream f_knn("test/t8knn.txt", std::ofstream::out | std::ofstream::app);
    //f_knn << (finish_knn - start_knn) << std::endl;

    std::vector<std::vector<float>> matrix_similarity(points.size(), std::vector<float>(points.size(), 0));

    double start_jaccard = omp_get_wtime();

#pragma omp parallel num_threads(n_threads)
    for (int i = 0; i < points.size(); i++) {
#pragma omp for schedule(dynamic, chunk_size)
        for (int j = 0; j < points.size(); j++) {
            if (i >= j) continue;

            matrix_similarity[i][j] = matrix_similarity[j][i] = calculate_jaccard(knn[i], knn[j]);
        }
    }

    double finish_jaccard = omp_get_wtime();
    printf("Jaccard in %.8f seconds\n", finish_jaccard - start_jaccard);
    //std::ofstream f_jaccard("test/dynamicINNER_chunk500.txt", std::ofstream::out | std::ofstream::app);
    //f_jaccard << (finish_jaccard - start_jaccard) << std::endl;

    int sample_size = 10;
    for (int i = 0; i < sample_size; i++) {
        for (int j = 0; j < sample_size; j++) {
            if (i == j) std::cout << "-\t";
            else printf("%.2f%%\t", matrix_similarity[i][j] * 100);
        }
        std::cout << std::endl;
    }

    return EXIT_SUCCESS;
}
