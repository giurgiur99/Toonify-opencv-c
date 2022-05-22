#include "stdafx.h"
#include "common.h"
#include <iostream>
#include <stdio.h>
#include <queue>
#include<stack>
#include <random>

using namespace std;

bool isInside1(Mat img, int i, int j) {

    int height = img.rows;
    int width = img.cols;

    if (i >= 0 && i < height) {
        if (j >= 0 && j < width) {
            return true;
        }
        else {
            return false;
        }
    }
    else {
        return false;
    }
}

// Ex.2: Function which generates a color image from a label matrix

Mat_<Vec3b> generateColors(int height, int width, Mat_<int> labels) {


   
    Mat_<Vec3b> dst(height, width);
    

    int maxLabel = 0;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (labels(i, j) > maxLabel) {
                maxLabel = labels(i, j);
            }
        }
    }

    default_random_engine gen;
    uniform_int_distribution<int> d(0, 255);

    std::vector<Vec3b> colors(maxLabel + 1);
    for (int i = 0; i <= maxLabel; i++) {
        
        uchar r = d(gen);
        uchar g = d(gen);
        uchar b = d(gen);

        colors.at(i) = Vec3b(r, g, b);
    }

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {

            int label = labels(i, j);

            if (label > 0)
                dst(i, j) = colors.at(labels(i, j));
            else
                dst(i, j) = Vec3b(255, 255, 255);
        }
    }


    return dst;
}


// Ex.1: BFS - Algorithm 1

void bfs() {

    Mat_<uchar> img = imread("ImagesLab5/diagonal.bmp", IMREAD_GRAYSCALE);
    int height = img.rows;
    int width = img.cols;

    // labels matrix
    Mat_<int> labels(height, width);
    labels.setTo(0); // initialize the matrix with 0


    // N4
    int di[4] = { -1, 0, 1, 0 };
    int dj[4] = { 0, -1, 0, 1 };

    // N8
    /*int di[8] = { -1, 0, 0, 1, -1, -1, 1, 1 };
    int dj[8] = { 0, -1, 1, 0, -1, 1, -1, 1 };*/

    int label = 0;


    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            
            if (img(i, j) == 0 && labels(i, j) == 0) {
            
                label++;
                std::queue<Point2i> Q;
                labels(i, j) = label;
                Q.push(Point2i(j, i));
                
                while (!Q.empty()) {
                
                    Point2i q = Q.front();
                    Q.pop();  


                    for (int k = 0; k < 4; k++) {
                        
                        if (isInside1(img, q.y + di[k], q.x + dj[k])) {
                            uchar neigh = img(q.y + di[k], q.x + dj[k]);

                            if (neigh == 0 && labels(q.y + di[k], q.x + dj[k]) == 0) {
                                labels(q.y + di[k], q.x + dj[k]) = label;
                                Q.push(Point2i(q.x + dj[k], q.y + di[k]));
                            }
                        }
                    }
                }
            }
        }
    }

    Mat_<Vec3b> coloredImg = generateColors(height, width, labels);

    imshow("Initial image", img);
    imshow("BFS", coloredImg);
    waitKey(0);
}

// Ex.3: Two-pass with equivalence classes - Algorithm 2
void twoPass() {

    Mat_<uchar> img = imread("ImagesLab5/diagonal.bmp", IMREAD_GRAYSCALE);
    int height = img.rows;
    int width = img.cols;

    // labels matrix
    Mat_<int> labels(height, width);
    labels.setTo(0); // initialize the matrix with 0

    int label = 0;

    // Np
    int di[4] = {0, -1, -1, -1};
    int dj[4] = {-1, -1, 0, 1};

    vector<vector<int>> edges(1000);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (img(i, j) == 0 && labels(i, j) == 0) {
                vector<int> L;
                for (int k = 0; k < 4; k++) {
                    if (isInside1(img, i + di[k], j + dj[k])) {
                        if (labels(i + di[k], j + dj[k]) > 0) {
                            L.push_back(labels(i + di[k], j + dj[k]));
                        }
                    }
                }
                if (L.size() == 0) {
                    label++;
                    labels(i, j) = label;
                }
                else {
                    int x = *min_element(L.begin(), L.end());
                    labels(i, j) = x;
                    for (int y : L) {
                        if (y != x) {
                            edges[x].push_back(y);
                            edges[y].push_back(x);
                        }
                    }
                }
            }
        }
    }

    Mat_<Vec3b> firstImg = generateColors(height, width, labels);
    imshow("Initial image", img);
    imshow("First pass", firstImg);

    int newLabel = 0;
    int* newLabels = new int[label + 1];
    // initialize the array with 0
    for (int i = 0; i <= label; i++) {
        newLabels[i] = 0;
    }

    for (int j = 1; j <= label; j++) {
        if (newLabels[j] == 0) {
            newLabel++;
            std::queue<int> Q;
            newLabels[j] = newLabel;
            Q.push(j);

            while (!Q.empty()) {
                int x = Q.front();
                Q.pop();
                for (int y : edges[x]) {
                    if (newLabels[y] == 0) {
                        newLabels[y] = newLabel;
                        Q.push(y);
                    }
                }
            }
        }
    }

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            labels(i, j) = newLabels[labels(i, j)];
        }
    }

    
    Mat_<Vec3b> secondImg = generateColors(height, width, labels);

    imshow("Second pass", secondImg);
    waitKey(0);
}


// Extra Ex.5: DFS algorithm
void dfs() {
    Mat_<uchar> img = imread("ImagesLab5/diagonal.bmp", IMREAD_GRAYSCALE);
    int height = img.rows;
    int width = img.cols;

    // labels matrix
    Mat_<int> labels(height, width);
    labels.setTo(0); // initialize the matrix with 0


    // N4
    //int di[4] = { -1, 0, 1, 0 };
    //int dj[4] = { 0, -1, 0, 1 };

    // N8
    int di[8] = { -1, 0, 0, 1, -1, -1, 1, 1 };
    int dj[8] = { 0, -1, 1, 0, -1, 1, -1, 1 };

    int label = 0;


    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {

            if (img(i, j) == 0 && labels(i, j) == 0) {

                label++;
                std::stack<Point2i> S;
                labels(i, j) = label;
                S.push(Point2i(j, i));

                while (!S.empty()) {

                    Point2i q = S.top();
                    S.pop();


                    for (int k = 0; k < 8; k++) {

                        if (isInside1(img, q.y + di[k], q.x + dj[k])) {
                            uchar neigh = img(q.y + di[k], q.x + dj[k]);

                            if (neigh == 0 && labels(q.y + di[k], q.x + dj[k]) == 0) {
                                labels(q.y + di[k], q.x + dj[k]) = label;
                                S.push(Point2i(q.x + dj[k], q.y + di[k]));
                            }
                        }
                    }
                }
            }
        }
    }

    Mat_<Vec3b> coloredImg = generateColors(height, width, labels);

    imshow("Initial image", img);
    imshow("DFS", coloredImg);
    waitKey(0);
}



void main() {
   bfs();
   twoPass();
   dfs();
}