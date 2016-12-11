#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <Eigen/Sparse>

#include <float.h>
#include <fstream>
#include <iostream>
#include <cstdio>
#include <boost/algorithm/string.hpp>
#include <vector>
#include <sstream>
#include "util.h"

using namespace cv;
using namespace std;



void update_database(){
    vector<Mat> views = load_data_base();
    vector<Mat> g = build_gabor();
    vector<vector<Mat> > R;
    vector<vector<vector<double> > > all_features;
    for(int i=0; i<views.size(); i++)
    {
        cout<<i<<" of "<<views.size()<<'\r' << flush;
        R.push_back(apply_gabor(views[i], g));
        all_features.push_back(compute_gabor_feature(R[i]));
    }
    cout << "Features of the database computed." << endl;
    // Chose the centroids and write their features in the file centroids.csv
    vocab(all_features);
    // Compute the frequencies of each centroids among all features (and write them in frequencies.csv)
    build_freq(all_features);
    // Compute and store the histograms of all training sketches in the file histograms.csv
    store_hist(all_features);
    cout << "Histograms and frequencies computed." << endl << "Training step done." << endl;
}

void test (String filename){
    Mat A=imread(filename);
    vector<Mat> g = build_gabor();
    vector<Mat> r = apply_gabor(A,g);
    vector<vector<double> > features = compute_gabor_feature(r);
    // Load the centroids
    vector<vector<double> > centroids = load_matrix("storage/centroids.csv");
    // Load the histograms of the database sketches
    vector<string> objects; // objects[i] is the name of the object represented on the sketch i
    vector<Eigen::SparseMatrix<double> > histograms = load_histograms("storage/histograms.csv", objects);
    // Load the frequencies of the centroids in the database
    vector<double> frequencies = load_matrix("storage/frequencies.csv")[0];
    cout << "All data loaded." << endl;
    // Constructs the histogram of the feature
    Eigen::SparseMatrix<double> hist = normalized_hist(centroids, features, frequencies);
    cout << "Histogram of the test image computed." << endl;
    // Compare hist with the histograms of the database and print the best matches
    compare_hist(histograms, hist, objects);
}

int main()
{
    update_database();
    test("test_image.png");

    return 0;
}
