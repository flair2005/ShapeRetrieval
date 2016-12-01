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
#include <set>

using namespace cv;
using namespace std;

typedef Eigen::Triplet<double> T;

Mat float2byte(const Mat& If)
{
    double minVal, maxVal;
    minMaxLoc(If,&minVal,&maxVal);
    Mat Ib;
    If.convertTo(Ib, CV_8U, 255.0/(maxVal - minVal),-minVal * 255.0/(maxVal - minVal));
    return Ib;
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

double euclideanDistance(vector<double> u, vector<double> v){
    int l = u.size();
    int ll = v.size();
    double dist = 0;
    if (l==ll){
        for(int i = 0;i<l;i++){
            dist += (u[i]-v[i])*(u[i]-v[i]);
        }
    }
    else{ cout<<"Could not measure distance between u and v :"<<endl<< "u :  ";
        for(int i = 0;i<l;i++)
            cout << u[i] << " ";
        cout << endl << "v :  ";
        for(int i = 0;i<ll;i++)
            cout << v[i] << " ";
        cout << endl;
    }
    return sqrt(dist);
}


vector<double> compute_hist (vector<vector<double> > centers, vector<vector<double> > test){
    int nc = centers.size();
    int nt = test.size();
    vector<double> res;
    for(int i = 0;i<nc;i++){
        res.push_back(0.);
    }
    for(int i = 0;i<nt;i++){
        int minIdx = 0;
        double minDist = euclideanDistance(centers[0],test[i]);
        for(int j = 1;j<nc;j++){
            double newDist = euclideanDistance(centers[j],test[i]);
            if (newDist < minDist){
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

Mat dft_inverse(Mat g)
{
    Mat padded;
    int m = getOptimalDFTSize(g.rows);

    copyMakeBorder(g, padded, 0, m - g.rows, 0, m - g.cols, BORDER_CONSTANT, Scalar::all(0)); // on the border add zero values

    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);      // Add to the expanded another plane with zeros

    //Mat complexI = g;
    dft(complexI, complexI, DFT_INVERSE);            // this way the result may fit in the source matrix
    //Mat magI = complexI;

    // compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    Mat magI = planes[0];

    magI += Scalar::all(1);                    // switch to logarithmic scale
    log(magI, magI);

    // crop the spectrum, if it has an odd number of rows or columns
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = magI.cols/2;
    int cy = magI.rows/2;

    Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

    Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);

    normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
    // viewable image form (float between values 0 and 1).

    return magI;
}

Mat fullDft(Mat M)
{
    Mat padded;
    int m = getOptimalDFTSize(M.rows);

    copyMakeBorder(M, padded, 0, m - M.rows, 0, m - M.cols, BORDER_CONSTANT, Scalar::all(0)); // on the border add zero values

    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);     // Add to the expanded another plane with zeros

    dft(complexI, complexI);            // this way the result may fit in the source matrix

    // compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    Mat magI = planes[0];

    magI += Scalar::all(1);                    // switch to logarithmic scale
    log(magI, magI);

    // crop the spectrum, if it has an odd number of rows or columns
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = magI.cols/2;
    int cy = magI.rows/2;

    Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

    Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);

    normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
    // viewable image form (float between values 0 and 1).

    return magI;
}

int main()
{
    Mat A=imread("plane.png");

    double w0 = 0.13; // peak-response frequency
    double sigmax = 0.02*0.05; // frequency bandwith : 0.02*A.cols / 0.02*0.13
    double sigmay = sigmax / 0.1; // angular bandwith : sigmax / 0.3
    //cout << sigmax << " ; " << sigmay << endl;
    int k = 4; // number of filter orientations
    double theta[k];
    int kernel_size = 1111;
    for(int i=0; i<k; i++)
    {
        theta[i] = CV_PI * i / k;
    }
    double u,v;
    Mat g[k];
    for(int i=0; i<k; i++)
    {
        g[i] = Mat(kernel_size, kernel_size, CV_32F);
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
                //cout << g[t].at<float>(i, j) << endl;
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

        //normalize(g[t], g[t], 0, 1, CV_MINMAX);
    }

    Mat I;
    Mat dftI;
    Mat R[k];
    for(int i=0; i<k; i++)
    {
        R[i] = g[i].clone();
    }
    cvtColor(A,I,CV_BGR2GRAY);
    I.convertTo(I, CV_32F);

    // Gabor à la main
    dft(I, dftI);
    for(int i=0; i<k; i++)
    {
        mulSpectrums(R[i], dftI, R[i], 0);
        dft(R[i], R[i], DFT_INVERSE);
        normalize(R[i], R[i], 0, 1, CV_MINMAX);
        /*imshow("Display", float2byte(R[i]));
        waitKey();*/
    }

    // Gabor par filter2D
    /*for(int i=0; i<k; i++)
    {
        //dft(g[i], g[i], DFT_INVERSE + DFT_SCALE);
        g[i] = dft_inverse(g[i]);
        //flip(g[i], g[i], -1);
        filter2D(I, R[i], CV_32F, g[i], Point(kernel_size - (kernel_size-1)/2, kernel_size - (kernel_size-1)/2), 0, BORDER_REPLICATE);
        imshow("Display2", float2byte(R[i]));
        waitKey();
    }*/


    /*// Gabor using getGaborKernel
    Mat Rb;
    int kernel_sizeb = 399;
    double sigma = 0.02*A.cols, thetab = CV_PI/4, lambda = 1./0.13, gamma = 0.3, psi = 0; // lambda = 0.07
    cv::Mat kernel = cv::getGaborKernel(cv::Size(kernel_sizeb,kernel_sizeb), sigma, thetab, lambda, gamma, psi);
    normalize(kernel, kernel, 0, 1, CV_MINMAX);
    cv::filter2D(I, Rb, CV_32F, kernel);
    imshow("Display", float2byte(kernel));
    waitKey();*/

    /*
    * Compute each Gabor feature
    */
    int nb_features = 1024; // 32*32
    int nb_tiles = 4;
    double feature_size = 0.2;
    int feature_dim = k*nb_tiles*nb_tiles; // dimension of one feature vector
    int local_patch_side = (int) floor(A.cols * sqrt(feature_size)); // =sqrt(area_image * feature_size)
    int gap = (int) floor((A.cols - local_patch_side) / 31); // gap between two key points on the image (we need to put 32 key points evenly
    // distributed on a row of length A.cols and with a margin of local_patch_side/2 on the link and right
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
    //printf("feature_dim=%d ; local_patch_side=%d ; gap=%d ; semi_side=%d ; pixel_per_cell=%d", feature_dim, local_patch_side, gap, semi_side, pixel_per_cell);

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


    /*
    * Load the centroïds
    */
    vector<vector<double> > centroids = load_centroids("centroids.csv");


    /*
    * Load the histograms of the database sketches
    */
    vector<string> objects; // objects[i] is the name of the object represented on the sketch i
    vector<Eigen::SparseMatrix<double> > histograms = load_histograms("histograms.csv", objects);


    /*
    * Load the frequencies of the centroids in the database
    */
    vector<double> frequencies = load_frequencies("frequencies.csv");


    /*
    * Constructs the histogram of the feature
    */
    int nb_views_per_model = 100;
    int nb_models = 1814;
    int N = nb_views_per_model * nb_models;
    vector<double> nearest_centroids = compute_hist(centroids, features);
    Eigen::SparseMatrix<double> hist(nearest_centroids.size(), 1);
    vector<T> tripletList;
    for(int i=0; i<nearest_centroids.size(); i++)
        tripletList.push_back(T(i, 0, nearest_centroids[i] / (float) nb_features * log((float) N / frequencies[i])));
    hist.setFromTriplets(tripletList.begin(), tripletList.end());


    /*
    * Compare hist with the histograms of the database
    */
    vector<double> dist; // distances between hist and the histograms of the database
    Eigen::SparseMatrix<double> product;
    for(int i=0; i<histograms.size(); i++)
    {
        product = hist * histograms[i].transpose();
        dist.push_back(product.coeffRef(0,0) / (hist.norm() * histograms[i].norm()));
    }
    vector<int> sorted_idx = min_indices(dist, 19);

    cout << "Best matches : ";
    for(int i=0; i<19; i++)
        cout << objects[sorted_idx[i]];
    cout << endl;

    return 0;
}
