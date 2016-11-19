#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

#include <iostream>

using namespace cv;
using namespace std;

Mat float2byte(const Mat& If)
{
    double minVal, maxVal;
    minMaxLoc(If,&minVal,&maxVal);
    Mat Ib;
    If.convertTo(Ib, CV_8U, 255.0/(maxVal - minVal),-minVal * 255.0/(maxVal - minVal));
    return Ib;
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
        imshow("Display", float2byte(R[i]));
        waitKey();
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
    int gap = (A.cols - local_patch_side) / 31; // gap between two key points on the image (we need to put 32 key points evenly
    // distributed on a row of length A.cols and with a margin of local_patch_side/2 on the link and right
    int semi_side = (int) floor(local_patch_side/2);
    int pixel_per_cell = (int) floor(local_patch_side / nb_tiles); // there are pixel_per_cell² pixels in one cell Cst
    double features[nb_features][feature_dim];
    for(int i=0; i<nb_features; i++)
    {
        memset(features[i], 0, feature_dim);
    }

    int countFeature = 0;
    int countdim = 0;
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
                            }
                        }
                        countdim++;
                    }
                }
            }
            countdim = 0;
            countFeature++;
        }
    }

    return 0;
}
