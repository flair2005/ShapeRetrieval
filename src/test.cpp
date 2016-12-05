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

using namespace cv;
using namespace std;

const double w0 = 0.13; // peak-response frequency
const double sigmax = 0.02*0.05; // frequency bandwith : 0.02*A.cols / 0.02*0.13
const double sigmay = sigmax / 0.1; // angular bandwith : sigmax / 0.3
const int kernel_size = 1111;
const int k = 4; // number of filter orientations
double theta[k];

const int nb_features = 1024; // 32*32
const int nb_tiles = 4;
const double feature_size = 0.2;
const int feature_dim = k*nb_tiles*nb_tiles; // dimension of one feature vector

const int nb_views_per_model = 12; //100
const int nb_models = 2; //1815
const int N = nb_views_per_model * nb_models;

typedef Eigen::Triplet<double> T;

Mat float2byte(const Mat& If)
{
    double minVal, maxVal;
    minMaxLoc(If,&minVal,&maxVal);
    Mat Ib;
    If.convertTo(Ib, CV_8U, 255.0/(maxVal - minVal),-minVal * 255.0/(maxVal - minVal));
    return Ib;
}

/*
* Build an array of Mat, each containing a Gabor filter
*/
vector<Mat> build_gabor()
{
    double u,v;
    vector<Mat> g;
    for(int i=0; i<k; i++)
    {
        g.push_back(Mat(kernel_size, kernel_size, CV_32F));
    }
    for(int t=0; t<k; t++)
    {
        for(int i=0; i<kernel_size; i++)
        {
            for(int j=0; j<kernel_size; j++)
            {
                u = cos(theta[t])*(-(double)ceil(kernel_size/2)+i) - sin(theta[t])*(-(double)ceil(kernel_size/2)+j);
                v = sin(theta[t])*(-(double)ceil(kernel_size/2)+i) + cos(theta[t])*(-(double)ceil(kernel_size/2)+j);
                g[t].at<float>(i, j) = exp(-2*CV_PI*CV_PI * ((u-w0)*(u-w0) * sigmax*sigmax + v*v * sigmay*sigmay));
            }
        }
        /*imshow("g", float2byte(g[t]));
        waitKey(0);*/

        // rearrange the quadrants of Fourier image  so that the origin is at the image center
        int cx = g[t].cols/2;
        int cy = g[t].rows/2;

        Mat q0(g[t], Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
        Mat q1(g[t], Rect(cx, 0, cx, cy));  // Top-Right
        Mat q2(g[t], Rect(0, cy, cx, cy));  // Bottom-Left
        Mat q3(g[t], Rect(cx, cy, cx, cy)); // Bottom-Right

        Mat tmp; // swap quadrants (Top-Left with Bottom-Right)
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);

        q1.copyTo(tmp); // swap quadrant (Top-Right with Bottom-Left)
        q2.copyTo(q1);
        tmp.copyTo(q2);
    }

    return g;
}


/*
* Apply the Gabor filter to the image A and return the k response images
*/
vector<Mat> apply_gabor(const Mat &A, const vector<Mat> &g)
{
    Mat I;
    Mat dftI;
    vector<Mat> R;
    for(int i=0; i<k; i++)
    {
        R.push_back(g[i].clone());
    }
    cvtColor(A,I,CV_BGR2GRAY);
    I.convertTo(I, CV_32F);

    dft(I, dftI);
    for(int i=0; i<k; i++)
    {
        //cout << "types: " << dftI.type() << " ; " << g[i].type() << endl;
        //cout << "sizes: " << dftI.size() << " ; " << g[i].size() << endl;
        mulSpectrums(R[i], dftI, R[i], 0);
        dft(R[i], R[i], DFT_INVERSE);
        normalize(R[i], R[i], 0, 1, CV_MINMAX);
        /*imshow("Display", float2byte(R[i]));
        waitKey();*/
    }

    return R;
}


/*
* Compute the Gabor features of one image
*/
vector<vector<double> > compute_gabor_feature(const vector<Mat> &R)
{
    int local_patch_side = (int) floor(R[0].cols * sqrt(feature_size)); // =sqrt(area_image * feature_size)
    int gap = (int) floor((R[0].cols - local_patch_side) / 31); // gap between two key points on the image (we need to put 32 key points evenly
    // distributed on a row of length R.cols and with a margin of local_patch_side/2 on the link and right
    int semi_side = (int) floor(local_patch_side/2);
    int pixel_per_cell = (int) floor(local_patch_side / nb_tiles); // there are pixel_per_cell² pixels in one cell Cst
    vector<vector<double> > features;
    for(int i=0; i<nb_features; i++)
    {
        features.push_back(vector<double>(feature_dim));
        for(int j=0; j<feature_dim; j++)
        {
            features[i][j] = 0;
        }
    }

    int countFeature = 0;
    int countdim = 0;
    double norm = 0; // norm of one feature, used to normalize each feature
    for(int i=0; i<32; i++)
    {
        for(int j=0; j<32; j++)
        {
            // compute the feature associated with the key point with coordinates (i, j)
            for(int th=0; th<k; th++)
            {
                // explore the nb_tiles*nb_tiles in the local patch
                for(int s=-nb_tiles/2; s<nb_tiles/2; s++)
                {
                    for(int t=-nb_tiles/2; t<nb_tiles/2; t++)
                    {
                        for(int x=0; x<pixel_per_cell; x++)
                        {
                            for(int y=0; y<pixel_per_cell; y++)
                            {
                                features[countFeature][countdim] += R[th].at<float>(semi_side + i*gap + s*pixel_per_cell + x,
                                                                    semi_side + j*gap + t*pixel_per_cell + y);
                                norm += features[countFeature][countdim] * features[countFeature][countdim];
                            }
                        }
                        countdim++;
                    }
                }
            }
            norm = sqrt(norm);
            for(int c=0; c<feature_dim; c++)
            {
                features[countFeature][c] /= norm; // normalize the feature
            }
            norm = 0;
            countdim = 0;
            countFeature++;
        }
    }

    return features;
}

vector<vector<double> > load_centroids(string filename)
{
    vector<vector<double> > centroids;
    string line;
    ifstream file(filename.c_str());
    if(file)
    {
        while(getline(file, line))
        {
            vector<string> s;
            boost::split(s, line, boost::is_any_of(";"));
            if(s[s.size()-1] == "")
                s.pop_back();

            vector<double> f;
            for(int i=0; i<s.size(); i++)
                f.push_back(atof(s[i].c_str()));
            centroids.push_back(f);
        }
    }
    else
        cout << "ERREUR: Impossible d'ouvrir le fichier." << endl;

    return centroids;
}

vector<Eigen::SparseMatrix<double> > load_histograms(string filename, vector<string> &objects)
{
    vector<Eigen::SparseMatrix<double> > histograms; // contains all the histograms of the training sketches
    int nb_centroids = 2500;
    string line;
    ifstream file(filename.c_str());
    if(file)
    {
        while(getline(file, line))
        {
            vector<string> s;
            boost::split(s, line, boost::is_any_of(";"));
            if(s[s.size()-1] == "")
                s.pop_back();
            objects.push_back(s.back());
            s.pop_back();

            Eigen::SparseMatrix<double> h(nb_centroids, 1);
            vector<T> tripletList;
            for(int i=0; i<s.size(); i++)
                tripletList.push_back(T(i,0,atof(s[i].c_str())));
            h.setFromTriplets(tripletList.begin(), tripletList.end());
            histograms.push_back(h);
        }
    }
    else
        cout << "ERREUR: Impossible d'ouvrir le fichier." << endl;

    return histograms;
}

double euclideanDistance(const vector<double> &u, const vector<double> &v)
{
    int l = u.size();
    int ll = v.size();
    double dist = 0;
    if (l==ll)
    {
        for(int i = 0; i<l; i++)
        {
            dist += (u[i]-v[i])*(u[i]-v[i]);
        }
    }
    else
    {
        cout<<"Could not measure distance between u and v :"<<endl<< "u :  ";
        for(int i = 0; i<l; i++)
            cout << u[i] << " ";
        cout << endl << "v :  ";
        for(int i = 0; i<ll; i++)
            cout << v[i] << " ";
        cout << endl;
    }
    return sqrt(dist);
}


vector<double> compute_hist (const vector<vector<double> > &centers, const vector<vector<double> > &test)
{
    int nc = centers.size();
    int nt = test.size();
    vector<double> res;
    for(int i = 0; i<nc; i++)
    {
        res.push_back(0.);
    }
    for(int i = 0; i<nt; i++)
    {
        int minIdx = 0;
        double minDist = euclideanDistance(centers[0],test[i]);
        for(int j = 1; j<nc; j++)
        {
            double newDist = euclideanDistance(centers[j],test[i]);
            if (newDist < minDist)
            {
                minDist = newDist;
                minIdx = j;
            }
        }
        res[minIdx]+= 1.;
    }
    return res;
}

vector<double> load_frequencies(string filename)
{
    vector<double> frequencies;
    string line;
    ifstream file(filename.c_str());
    if(file)
    {
        getline(file, line);
        vector<string> s;
        boost::split(s, line, boost::is_any_of(";"));
        if(s[s.size()-1] == "")
            s.pop_back();

        for(int i=0; i<s.size(); i++)
            frequencies.push_back(atof(s[i].c_str()));
    }
    else
        cout << "ERREUR: Impossible d'ouvrir le fichier." << endl;

    return frequencies;
}


/*
* Constructs the histogram of the feature
*/
Eigen::SparseMatrix<double> buil_hist(const vector<vector<double> > &centroids, const vector<vector<double> > &features, const vector<double> &frequencies)
{
    vector<double> nearest_centroids = compute_hist(centroids, features);
    Eigen::SparseMatrix<double> hist(nearest_centroids.size(), 1);
    vector<T> tripletList;
    for(int i=0; i<nearest_centroids.size(); i++)
    {
        if(nearest_centroids[i] != 0)
            tripletList.push_back(T(i, 0, nearest_centroids[i] / (float) nb_features * log((float) N / frequencies[i])));
    }
    hist.setFromTriplets(tripletList.begin(), tripletList.end());

    return hist;
}


/*
* Gives the indices of the k smallest elements of tab. Complexity: k*tab.size()
*/
vector<int> min_indices(vector<double> &tab, int k)
{
    vector<int> indices;
    int current;
    double min_dist;
    int n = tab.size();
    for(int i=0; i<k; i++)
    {
        current = 0;
        min_dist = tab[0];
        for(int j=1; j<n; j++)
        {
            if(tab[j] < min_dist)
            {
                current = j;
                min_dist = tab[j];
            }
        }
        indices.push_back(current);
        tab[current] = DBL_MAX;
    }
    return indices;
}


/*
* Compare hist with the histograms of the database and print the best matches
*/
void compare_hist(const vector<Eigen::SparseMatrix<double> > &histograms, const Eigen::SparseMatrix<double> &hist, const vector<string> &objects)
{
    vector<double> dist; // distances between hist and the histograms of the database
    Eigen::SparseMatrix<double> product;
    //cout << hist << endl;
    for(int i=0; i<histograms.size(); i++)
    {
        product = hist * histograms[i].transpose();
        cout << product << endl;
        dist.push_back(product.coeffRef(0,0) / (hist.norm() * histograms[i].norm()));
    }
    vector<int> sorted_idx = min_indices(dist, 5);

    cout << "Best matches : ";
    for(int i=0; i<5; i++)
        cout << objects[sorted_idx[i]]<< " ; ";
    cout << endl;

    /*for(int i=0; i<objects.size(); i++)
    {
        cout << dist[i] << endl;
    }*/
}


int main()
{
    Mat A=imread("test_image.png");

    // Initialize the orientations theta
    for(int i=0; i<k; i++)
    {
        theta[i] = CV_PI * i / k;
    }

    // Build an array of Mat, each containing a Gabor filter
    vector<Mat> g = build_gabor(); // array of size k

    // Apply the Gabor filter to the image A and return the k response images
    vector<Mat> R = apply_gabor(A, g); // array of size k
    cout << "Filtered image computed." << endl;

    // Compute each Gabor feature
    vector<vector<double> > features = compute_gabor_feature(R);
    cout << "Features of test image computed." << endl;

    // Load the centroïds
    vector<vector<double> > centroids = load_centroids("storage/centroids.csv");

    // Load the histograms of the database sketches
    vector<string> objects; // objects[i] is the name of the object represented on the sketch i
    vector<Eigen::SparseMatrix<double> > histograms = load_histograms("storage/histograms.csv", objects);

    // Load the frequencies of the centroids in the database
    vector<double> frequencies = load_frequencies("storage/frequencies.csv");
    cout << "All data loaded." << endl;

    // Constructs the histogram of the feature
    Eigen::SparseMatrix<double> hist = buil_hist(centroids, features, frequencies);
    cout << "Histogram of the test image computed." << endl;

    // Compare hist with the histograms of the database and print the best matches
    compare_hist(histograms, hist, objects);

    return 0;
}
