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

#define SSTR( x ) static_cast< std::ostringstream & >( \
        ( std::ostringstream() << std::dec << x ) ).str()

using namespace cv;
using namespace std;

typedef Eigen::Triplet<double> T;

vector<Mat> build_gabor();
vector<Mat> apply_gabor(const Mat &, const vector<Mat> &);
vector<vector<double> > compute_gabor_feature(const vector<Mat> &);
vector<vector<double> > features(Mat, const vector<Mat> &);

vector<vector<double> > load_matrix(string);
vector<Mat> load_data_base();
vector<Eigen::SparseMatrix<double> > load_histograms(string, vector<string> &);

void vocab (vector<vector<vector<double> > >);
double euclideanDistance(const vector<double> &, const vector<double> &);
vector<double> raw_hist (const vector<vector<double> > &, const vector<vector<double> > &);
void build_freq(vector<vector<vector<double> > > test_base);
Eigen::SparseMatrix<double> normalized_hist(const vector<vector<double> > &, const vector<vector<double> > &, const vector<double> &);
void store_hist(vector<vector<vector<double> > >);

vector<int> max_indices(vector<double>, int );
void compare_hist(const vector<Eigen::SparseMatrix<double> > &, const Eigen::SparseMatrix<double> &, const vector<string> &);
