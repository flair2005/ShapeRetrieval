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

void make_csv(const vector<double> & v, const String & filename){
    String nfn = filename+".csv";
    ofstream myStream(nfn.c_str());
    if(myStream){
        for (int r = 0;r<v.size();r++){
            myStream<<v[r]<<';';
        }
        myStream<<endl;
    } else {
    cout<<"fuck you all beatches !! (écriture dans fichier "<<filename<<" impossible)";
    }
}

void vocab (vector<vector<vector<double> > > (test_base) , int k){//TODO allow for more supple matrix sizes
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
    Mat labels;// performing k-means on all features to keep k centroids as the vocabulary
    Mat centers;
    kmeans(reduced, k, labels, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers );

    // écrire centers dans fichier centroids.csv;
    ofstream myStream("centroids.csv");
    if(myStream){
        for (int r = 0;r<centers.rows;r++){
            for(int c = 0;c<centers.cols;c++){
                myStream<<centers.at<float>(r,c)<<';';
            }
            myStream<<endl;
        }
    } else {
    cout<<"fuck you all beatches !! (écriture dans fichier centroids impossible)";
    }
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
          //  cout<<"foudaline !"<<endl;
            vector<string> s;
            boost::split(s, line, boost::is_any_of(";"));
            if(s[s.size()-1] == "")
                s.pop_back();

            vector<double> f;
            for(int i=0; i<s.size(); i++){
                f.push_back(atof(s[i].c_str()));
                //cout<<atof(s[i].c_str())<<endl;
            }
            centroids.push_back(f);
        }
    }
    else
        cout << "ERREUR: Impossible d'ouvrir le fichier centroids." << endl;

    return centroids;
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
    make_csv(res,"histograms");
    return res;
}

void frequency(vector<vector<vector<double> > > test_base){
    int n_sketches = test_base.size();
    int fv_size = test_base[0].size();
    int dim = test_base[0][0].size();
    vector<vector<double> > centers = load_centroids("centroids.csv");
    vector<vector<double> > test;
    for(int i = 0;i<n_sketches;i++){
        for(int j = 0;j<fv_size;j++){
            vector<double> feature;
            for(int k = 0;k<dim;k++){
                feature.push_back(test_base[i][j][k]);
            }
            test.push_back(feature);
        }
    }
    vector<double> hist = compute_hist(centers,test);
    double sum = 0.;
    for(int i = 0;i<hist.size();i++){
        sum+=hist[i];
    }

    ofstream myStream("frequencies.csv");
    if(myStream){
        for (int r = 0;r<hist.size();r++){

            myStream<<hist[r]/sum << ';';
        }
    } else {
    cout<<"fuck you all beatches !! (écriture dans fichier frequencies impossible)";
    }

}

vector<vector<double> > rmg(int M,int N){
    vector<vector<double> > res;
    //cout<<"generating random "<<M<<"x"<<N<<" matrix"<<endl;
    for(int i = 0;i<M&&i<10;i++){
        //cout<<"   line "<<i<<" of "<<M<<" : "<<endl;
        vector<double> buf;
        for(int j = 0;j<N;j++){
            buf.push_back(((double)rand()) / RAND_MAX);
            //cout<<" "<<j;
        }
        //cout<<"."<<endl;
        res.push_back(buf);
    }
    return res;
}



vector<vector<vector<double> > > rtbg(int n_features,int dim_feature,int n_testcases){
    //cout<<"generating random test base of "<<O<<" "<<M<<"x"<<N<<" matrices"<<endl;
    vector<vector<vector<double> > > res;
    for(int i = 0;i<n_testcases;i++)
        res.push_back(rmg(n_features,dim_feature));
    return res;
}


void print(vector<vector<double> > m){
    for(int i = 0;i<m.size();i++){
        cout<<endl;
        for(int j = 0;j<m[0].size();j++){
            cout<<' '<<m[i][j];
        }
    }
}

void print(vector<vector<vector<double> > > tb){
    cout<<"/---- Your test suite ----\\";
    for(int i = 0;i<tb.size();i++){
        print(tb[i]);
        cout<<endl;
    }
    cout<<"\\---- etius tset ruoY ----/"<<endl;
}

int main()
{
    vector<vector<vector<double> > > test_base = rtbg(100,40,40);
    print(test_base);
    vocab(test_base,5);
    frequency(test_base);
    vector<vector<vector<double> > > test_base2 = rtbg(100,40,40);
    print(test_base2);

    vector<vector<double> > features;
    for(int i = 0;i<test_base2.size();i++)
        for(int j = 0;j<test_base2[i].size();j++){
            features.push_back(test_base2[i][j]);
        }

    vector<vector<double> > centroids = load_centroids("centroids.csv");

    vector<string> objects; // objects[i] is the name of the object represented on the sketch i
    vector<Eigen::SparseMatrix<double> > histograms = load_histograms("histograms.csv", objects);
    vector<double> frequencies = load_frequencies("frequencies.csv");
 //   int nb_views_per_model = 100;
 //   int nb_models = 1814;
    int N = 10;//nb_views_per_model * nb_models;
    vector<double> nearest_centroids = compute_hist(centroids, features);
    Eigen::SparseMatrix<double> hist(nearest_centroids.size(), 1);
    vector<T> tripletList;
    for(int i=0; i<nearest_centroids.size(); i++)
        tripletList.push_back(T(i, 0, nearest_centroids[i] / (float) 5 * log((float) N / frequencies[i])));
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
    vector<int> sorted_idx = min_indices(dist, 3);

    cout << "Best matches : ";
    for(int i=0; i<3; i++)
        cout << objects[sorted_idx[i]]<<" ";
    cout << endl;
}


