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

const double w0 = 0.13; // peak-response frequency
const double sigmax = 0.02*0.05; // frequency bandwith : 0.02*A.cols / 0.02*0.13
const double sigmay = sigmax / 0.1; // angular bandwith : sigmax / 0.3
const int kernel_size = 1111;
const int k = 4; // number of filter orientations

const int nb_features = 1024; // 32*32
const int nb_tiles = 4;
const double feature_size = 0.2;
const int feature_dim = k*nb_tiles*nb_tiles; // dimension of one feature vector

const int nb_views_per_model = 12; //100
const int nb_models = 2; //1815
const int N = nb_views_per_model * nb_models;

const int vocabulary_size = 100; // number of centroids: 2500

typedef Eigen::Triplet<double> T;

/**
Functions to turn an image into a features vector
*/

/*
* Build an array of Mat, each containing a Gabor filter
*/
vector<Mat> build_gabor()
{
    double theta[k];
    for(int i=0; i<k; i++)
    {
        theta[i] = CV_PI * i / k;
    }
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

/* big sum up
*/

vector<vector<double> > features(Mat A, const vector<Mat> &g){
    // Apply the Gabor filter to the image A and return the k response images
    vector<Mat> R = apply_gabor(A, g); // array of size k
    cout << "Filtered image computed." << endl;

    // Compute each Gabor feature
    vector<vector<double> > features = compute_gabor_feature(R);
    cout << "Features of test image computed." << endl;
}

/**
to read a csv file, to load sketches
*/

vector<vector<double> > load_matrix(string filename)
{
    vector<vector<double> > res;
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
            res.push_back(f);
        }
    }
    else
        cout << "ERREUR: Impossible d'ouvrir le fichier." << endl;

    return res;
}

vector<Mat> load_data_base(){
    // Load all the training sketches
    // They should be named mxxx-yyy.png where xxx is the number of the model and yyy the number of the sampled view
    vector<Mat> views;
    for(int m=0; m<nb_models; m++)
    {
        for(int i=0; i<nb_views_per_model; i++)
        {
            views.push_back(imread("train_data/m"+SSTR(m)+"-"+SSTR(i)+".png"));
        }
    }
    cout << "Data loaded." << endl;
    return views;
}

vector<Eigen::SparseMatrix<double> > load_histograms(string filename, vector<string> &objects)
{
    vector<Eigen::SparseMatrix<double> > histograms; // contains all the histograms of the training sketches
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

            Eigen::SparseMatrix<double> h(vocabulary_size, 1);
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

/**
Functions to create centroids, frequencies and histograms vectors
*/

/*
* Compute the centroids and store their Gabor features in the file centroids.csv
*/
void vocab (vector<vector<vector<double> > > test_base){
    int n_sketches = test_base.size();
    int reduced_fv_size = 10;
    int fv_size = test_base[0].size();
    int dim = test_base[0][0].size();
    int attempts = 5;// attempts of k-means

    Mat reduced(n_sketches*reduced_fv_size,dim,CV_32F);// reducing number of features of each sketch
    for(int i = 0;i<n_sketches;i++){
        for(int j = 0;j<reduced_fv_size;j++){
            int next = rand()%fv_size;
            for(int l = 0;l<dim;l++)
                reduced.at<float>(i*reduced_fv_size+j,l)=test_base[i][next][l];
        }
    }
    Mat labels;// performing k-means on all features to keep vocabulary_size centroids as the vocabulary
    Mat centers;
    kmeans(reduced, vocabulary_size, labels, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);

    ofstream myStream("storage/centroids.csv");
    if(myStream){
        for (int r = 0;r<centers.rows;r++){
            for(int c = 0;c<centers.cols;c++){
                myStream<<centers.at<float>(r,c)<<';';
            }
            if(r<centers.rows-1)
                myStream<<endl;
        }
    } else {
    cout<<"Écriture dans le fichier centroids.csv impossible";
    }
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

vector<double> raw_hist (const vector<vector<double> > &centers, const vector<vector<double> > &test)
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

/*
* Build the file frequencies.csv (containing the frequency of each centroid among all features)
*/
void build_freq(vector<vector<vector<double> > > test_base){
    int n_sketches = test_base.size();
    int fv_size = test_base[0].size();
    int dim = test_base[0][0].size();
    int m, view;
    vector<vector<double> > centers = load_matrix("storage/centroids.csv");
    vector<double> hist;
    for(int i = 0;i<n_sketches;i++){
            vector<double> temp = raw_hist(centers,test_base[i]);
            for(int j = 0;j<temp.size();j++){
                if(i==0)
                    hist.push_back(temp[j]);
                else
                    hist[j]+=temp[j];
            }
        }

    double sum = 0.;
    for(int i = 0;i<hist.size();i++){
        sum+=hist[i];
    }

    ofstream myStream("storage/frequencies.csv");
    if(myStream){
        for (int r = 0;r<hist.size();r++){

            myStream<<hist[r]/sum << ';';
        }
    } else {
    cout<<"Écriture dans le fichier frequencies.csv impossible";
    }

}

Eigen::SparseMatrix<double> normalized_hist(const vector<vector<double> > &centroids, const vector<vector<double> > &features, const vector<double> &frequencies)
{
    vector<double> nearest_centroids = raw_hist(centroids, features);
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
* Build the files histograms.csv (containing the histograms of every sketch of the database)
*/
void store_hist(vector<vector<vector<double> > > test_base){
    int n_sketches = test_base.size();
    int fv_size = test_base[0].size();
    int dim = test_base[0][0].size();
    int m, view;
    vector<vector<double> > centers = load_matrix("storage/centroids.csv");
    vector<double> frequencies = load_matrix("storage/frequencies.csv")[0];
    vector<double> hist;
    ofstream streamHist("storage/histograms.csv");
    if(streamHist)
    {
        for(int i = 0;i<n_sketches;i++){
            vector<double> temp = raw_hist(centers,test_base[i]);
            for(int j = 0;j<temp.size();j++){
                if(i==0)
                    hist.push_back(temp[j]);
                else
                    hist[j]+=temp[j];
            }
            for(int r=0; r<temp.size(); r++)
            {
                temp[r] = temp[r] / (float) nb_features * log((float) N / frequencies[i]);
                streamHist << temp[r] << ";";
            }
            m = i / nb_views_per_model; // quotient of the euclidean division
            view = i - m*nb_views_per_model; // remainder of the euclidean division
            streamHist << "m"+SSTR(m)+"-"+SSTR(view); // add the name of the object at the end of the histogram
            if(i<n_sketches-1)
                streamHist << endl;
        }
    }
    else {
        cout<<"Écriture dans le fichier histograms.csv impossible";
    }
}

/**
Functions to find best models
*/

/*
* Gives the indices of the k greatest elements of tab. Complexity: k*tab.size()
*/
vector<int> max_indices(vector<double> tab, int k)
{
    vector<int> indices;
    int current;
    double max_dist;
    int n = tab.size();
    for(int i=0; i<k; i++)
    {
        current = 0;
        max_dist = tab[0];
        for(int j=1; j<n; j++)
        {
            if(tab[j] > max_dist)
            {
                current = j;
                max_dist = tab[j];
            }
        }
        indices.push_back(current);
        tab[current] = DBL_MIN;
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
    for(int i=0; i<histograms.size(); i++)
    {
        product = hist.transpose() * histograms[i];
        dist.push_back(product.coeffRef(0,0) / (hist.norm() * histograms[i].norm()));
    }
    vector<int> sorted_idx = max_indices(dist, 5);

    cout << "Best matches : ";
    for(int i=0; i<5; i++)
        cout << objects[sorted_idx[i]]<< " ; ";
    cout << endl;
}
