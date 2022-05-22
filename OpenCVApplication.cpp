// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <random>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;

#define Pi 3.1415926

bool isInside(Mat img, int i, int j) {
	int height = img.rows;
	int width = img.cols;

	if (i > 0 && j > 0 && i < height && j < width) {
		return true;
	}
	else {
		return false;
	}
}

Mat_<uchar> generate_StructElem(int n, bool val) {

	Mat_<uchar> struct_elem(n, n, 255);

	if (val) {
		int middle = n / 2;
		for (int i = 0; i < n; i++) {
			struct_elem(i, middle) = 0;
			struct_elem(middle, i) = 0;
		}
	}
	else {
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				struct_elem(i, j) = 0;
			}
		}
	}

	return struct_elem;
}


Mat	medianFilter(Mat src) {
	Mat dst = src.clone();
	int w = 3;
	int h = 3;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			int sx = j - w + 1;
			int sy = i - h + 1;
			int ex = j + w - 1;
			int ey = i + h - 1;
			vector<int> x;
			vector<int> y;
			vector<int> z;
			Vec3b pixel;
			for (int ti = sy; ti <= ey; ti++) {
				for (int tj = sx; tj <= ex; tj++) {
					pixel = src.at<Vec3b>((ti + src.rows) % src.rows, (tj + src.cols) % src.cols);
					x.push_back(pixel[0]);
					y.push_back(pixel[1]);
					z.push_back(pixel[2]);
				}
			}
			sort(x.begin(), x.end());
			sort(y.begin(), y.end());
			sort(z.begin(), z.end());
			int pos = x.size() / 2;
			int mx = x[pos];
			int my = y[pos];
			int mz = z[pos];
			dst.at<Vec3b>(i, j) = Vec3b(mx, my, mz);
		}
	}
	return dst;

}



double Gaussian(double sigma, double x)
{
	return exp(-pow(x, 2) / (2 * pow(sigma, 2))) / (sigma * pow(2 * Pi, 0.5));
}

double getDistance(int x1, int y1, int x2, int y2) {
	return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
}


Vec3b getPix(Mat image, int y, int x) {
	int ny = (y + image.rows) % image.rows;
	int nx = (x + image.cols) % image.cols;
	return image.at<Vec3b>(ny, nx);
}

double GetWp(Mat image, int y, int x, double sigma1, double sigma2, int window, int k) {

	double wp = 0;
	int w = window / 2;

	for (int pi = y - w; pi < y + w + 1; pi++) {
		for (int pj = x - w; pj < x + w + 1; pj++) {
			double distance = getDistance(x, y, pj, pi);
			double lightdiff = abs(getPix(image, y, x)[k] - getPix(image, pi, pj)[k]);
			wp += Gaussian(sigma1, distance) * Gaussian(sigma2, lightdiff);
		}
	}

	return wp;
}


double GetWpI(Mat image, int y, int x, double sigma1, double sigma2, int window, int k) {

	double wp = 0;
	int w = window / 2;

	for (int pi = y - w; pi < y + w + 1; pi++) {
		for (int pj = x - w; pj < x + w + 1; pj++) {
			double distance = getDistance(x, y, pj, pi);
			double lightdiff = abs(getPix(image, y, x)[k] - getPix(image, pi, pj)[k]);
			wp += Gaussian(sigma1, distance) * Gaussian(sigma2, lightdiff) * getPix(image, pi, pj)[k];
		}
	}

	return wp;
}


Mat bilateralFilter(Mat src, double sigma1, double sigma2) {
	Mat dst = src.clone();
	int width = src.cols;
	int height = src.rows;
	int  window = 7;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {


			for (int k = 0; k < 3; k++) {
				double wp = GetWp(src, i, j, sigma1, sigma2, window, k);
				double sum = GetWpI(src, i, j, sigma1, sigma2, window, k);
				dst.at<Vec3b>(i, j)[k] = sum / wp;
			}

		}
	}
	return dst;
}


Mat dilation(Mat src, Mat struct_elem) {


	int height = src.rows;
	int width = src.cols;
	Mat dst(height, width, CV_8UC1);



	int structElem_i = struct_elem.rows / 2;
	int structElem_j = struct_elem.cols / 2;


	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (src.at<uchar>(i, j) == 0) {

				dst.at<uchar>(i, j) = 0;

				for (int a = 0; a < struct_elem.rows; a++) {
					for (int b = 0; b < struct_elem.cols; b++) {
						if (struct_elem.at<uchar>(a, b) == 0) {
							if (isInside(src, i + a - structElem_i, j + b - structElem_j)) {
								dst.at<uchar>(i + a - structElem_i, j + b - structElem_j) = 0;
							}
						}
					}
				}
			}
			else dst.at<uchar>(i, j) = 255;
		}
	}


	return dst;
}


Mat erosion(Mat src, Mat struct_elem) {

	int height = src.rows;
	int width = src.cols;
	Mat_<uchar> dst = src.clone();

	int structElem_i = struct_elem.rows / 2;
	int structElem_j = struct_elem.cols / 2;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (src.at<uchar>(i, j) == 0) {

				for (int a = 0; a < struct_elem.rows; a++) {
					for (int b = 0; b < struct_elem.cols; b++) {
						if (struct_elem.at<uchar>(a, b) == 0) {
							if (isInside(src, i + a - structElem_i, j + b - structElem_j)) {
								if (src.at<uchar>(i + a - structElem_i, j + b - structElem_j) == 255) {
									dst(i, j) = 255;
								}
							}
						}
					}
				}
			}
		}
	}
	return dst;
}


Mat TakeNegative(cv::Mat& source)
{
	Mat dst(source.size(), CV_8U);
	for (int r = 0; r < source.rows; r++)
	{
		for (int c = 0; c < source.cols; c++)
		{
			dst.at<uchar>(r, c) = (255 - source.at<uchar>(r, c));
		}
	}
	return dst;
}


void convolutionINT(Mat_<int>& filter, Mat_<uchar>& img, Mat_<int>& output)
{
	output.create(img.size());
	output.setTo(0);
	int kk = (filter.rows - 1) / 2;
	for (int i = kk; i < img.rows - kk; i++)
		for (int j = kk; j < img.cols - kk; j++)
		{
			float sum = 0;
			for (int k = 0; k < filter.rows; k++)
				for (int l = 0; l < filter.cols; l++)
					sum += img(i + k - kk, j + l - kk) * filter(k, l);
			//sum = min(abs(sum / scalingCoeff), 255);
			output(i, j) = sum;
		}

}

Mat Sobelx(Mat src)
{
	Mat_<int> m = (Mat_<int>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
	Mat_<int> ret;
	convolutionINT(m, (Mat_<uchar>)src, ret);
	return ret;
}

Mat Sobely(Mat src) {

	Mat_<int> m = (Mat_<int>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);
	Mat_<int> ret;
	convolutionINT(m, (Mat_<uchar>)src, ret);
	return ret;
}


Mat getEdges(Mat src)
{
;
	Mat_<uchar> dest = Mat(src.rows, src.cols, CV_8UC1, Scalar(0));
	Mat_<uchar> orientation = Mat(src.rows, src.cols, CV_8UC1, Scalar(0));
	Mat_<uchar> magnitude = Mat(src.rows, src.cols, CV_8UC1, Scalar(0));

	Mat_<int> x = Sobelx(src);
	Mat_<int> y = Sobely(src);


	int maxi = -500, mini = 500;
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
		{
			if (x.at<int>(i, j) < mini)
				mini = x.at<int>(i, j);
			if (x.at<int>(i, j) > maxi)
				maxi = x.at<int>(i, j);

		}


	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
		{
			uchar res = sqrt(x.at<int>(i, j) * x.at<int>(i, j) + y.at<int>(i, j) * y.at<int>(i, j)) / (4 * sqrt(2));
			magnitude.at<uchar>(i, j) = res;
			double at = atan2(y.at<int>(i, j), x.at<int>(i, j));
			if (at < 0)
				at = at + PI;
			if (at < PI / 8 || at >= 7 * PI / 8)
				orientation.at<uchar>(i, j) = 1;
			else if (at >= PI / 8 && at < 3 * PI / 8)
				orientation.at<uchar>(i, j) = 2;
			else if (at >= 3 * PI / 8 && at < 5 * PI / 8)
				orientation.at<uchar>(i, j) = 3;
			else if (at >= 5 * PI / 8 && at < 7 * PI / 8)
				orientation.at<uchar>(i, j) = 4;

		}

	for (int r = 1; r < src.rows - 1; r++)
		for (int c = 1; c < src.cols - 1; c++)
		{
			dest[r][c] = magnitude[r][c];
			switch (orientation[r][c])
			{
			case 1:
				if (magnitude[r][c] < magnitude[r][c + 1] || magnitude[r][c] < magnitude[r][c - 1])
					dest[r][c] = 0;
				break;
			case 2:
				if (magnitude[r][c] < magnitude[r - 1][c + 1] || magnitude[r][c] < magnitude[r + 1][c - 1])
					dest[r][c] = 0;
				break;
			case 3:
				if (magnitude[r][c] < magnitude[r - 1][c] || magnitude[r][c] < magnitude[r + 1][c])
					dest[r][c] = 0;
				break;
			case 4:
				if (magnitude[r][c] < magnitude[r + 1][c + 1] || magnitude[r][c] < magnitude[r - 1][c - 1])
					dest[r][c] = 0;
				break;
			}
		}
	return dest;
}


Mat adaptiveThresholding(Mat src, int percent)
{
	int threshold_high = 0, threshold_low;
	Mat_<uchar> magnitude = getEdges(src);
	Mat_<uchar> threeColourMat = Mat(src.rows, src.cols, CV_8UC1, Scalar(0));
	int histo[256];
	for (int i = 0; i <= 255; i++)
		histo[i] = 0;
	for (int i = 0; i < magnitude.rows; i++)
		for (int j = 0; j < magnitude.cols; j++)
			histo[(int)magnitude.at<uchar>(i, j)]++;
	int nonEdgePixels = (1 - percent / 100) * ((magnitude.rows - 2) * (magnitude.cols - 2) - histo[0]);
	int sum = 0;
	for (threshold_high = 1; threshold_high <= 255 && sum <= nonEdgePixels; threshold_high++)
		sum += histo[threshold_high];
	threshold_low = 0.4 * threshold_high;
	for (int i = 0; i < magnitude.rows; i++)
		for (int j = 0; j < magnitude.cols; j++)
		{
			if (magnitude.at<uchar>(i, j) > threshold_high)
				threeColourMat.at<uchar>(i, j) = 255;
			else if (magnitude.at<uchar>(i, j) > threshold_low)
				threeColourMat.at<uchar>(i, j) = 128;
			else
				threeColourMat.at<uchar>(i, j) = 0;
		}
	return threeColourMat;
}




Mat recombine(cv::Mat& color, cv::Mat& edges, double factor) {
	cv::Vec3b colorpixel;
	cv::Mat final_image;
	color.copyTo(final_image);
	for (int r = 0; r < color.rows - 1; r++)
	{
		for (int c = 0; c < color.cols - 1; c++)
		{
			if (edges.at<uchar>(r, c)) {
				continue;
			}
			colorpixel = color.at<cv::Vec3b>(r, c);

			colorpixel[0] = (colorpixel[0] * factor); 
			colorpixel[1] = (colorpixel[1] * factor);
			colorpixel[2] = (colorpixel[2] * factor); 

			final_image.at<cv::Vec3b>(r, c) = colorpixel;
		}
	}
	return final_image;
}



int main()
{
	Mat img;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		img = imread(fname, IMREAD_COLOR);

		Mat filtered, filteredResized;
		Mat  imgResized;
		int x = 24, y = 6;
		cv::resize(img, imgResized, cv::Size(), 0.5, 0.5);

		filteredResized = bilateralFilter(imgResized, x, y);
		for (int i = 0; i < 5; i++) {
			filteredResized = bilateralFilter(filteredResized, x, y);
		}


		cv::resize(filteredResized, filtered, cv::Size(), 2.00, 2.00);
		filtered = medianFilter(filtered);

		Mat src(filtered.cols, filtered.rows, CV_8UC1);
		cvtColor(filtered, src, cv::COLOR_BGR2GRAY);
		Mat edges = adaptiveThresholding(src, 5);
		Mat myStructElem1 = generate_StructElem(1, true);
		Mat myStructElem2 = generate_StructElem(2, true);
		edges = dilation(edges, myStructElem1);
		edges = erosion(edges, myStructElem2);

		edges = TakeNegative(edges);

		imshow("Orginal", img);
		imshow("Strong edges", edges);
		imshow("Filtered", filtered);
		Mat final = recombine(filtered, edges, 0.5);
		imshow("Final", final);


		waitKey(0);
	}
	 
	return 0;
}