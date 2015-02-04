#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "opencv/cxmisc.h"
#include <math.h>
#include <stdio.h>
#include "msImageProcessor.h"
#include "edison_wrapper_mex.h"
#include "BgImage.h"
#include "BgEdge.h"
#include "BgEdgeList.h"
#include "BgEdgeDetect.h"
#include "../ANN/ANN.h"


#define SHARP_THRESHOLD 1.4375
#define CANNY_THRESHOLD_1 0.0270		
#define CANNY_THRESHOLD_2 0.0750
#define SIGMA_THRESHOLD 1.5
#define SPATIAL_BANDWITH 6
#define RANGE_BANDWITH 6
#define MINIMUM_SEGMANTATION_AREA 50
#define FILENAME "DSC_0027.jpg"
#define DISTANCE_FROM_CANNY 8

struct g {
	double gtheta;
	double magn_x;
	double magn_y;
	double theta;
};
struct lamda {
	int l;
	bool dir;// 0 for x, 1 for y
	int small_end;
	int big_end;
};
struct sigma {
	float s;
	bool dir;// 0 for x, 1 for y
};
struct laws {
	int nums[16];
};

int sobel_y[3][3] = { { 1, 0, -1 }, { 2, 0, -2 }, { 1, 0, -1 } };
int sobel_x[3][3] = { { 1, 2, 1 }, { 0, 0, 0 }, { -1, -2, -1 } };
int kernels1[14][5][5] = {
		{ { -1,	-4, -6, -4,	-1	},	{ -2, -8, -12, -8, -2 }, { 0, 0, 0, 0, 0 }, { 4, 16, 24, 16, 4 }, { 1, 4, 6, 4, 1 } },//E5L5 
		{ { -1,	-4, -6, -4,	-1	},	{ 0, 0, 0, 0, 0 }, { 2, 8, 12, 8, 2 }, { 0, 0, 0, 0, 0 }, { -1, -4, -6, -4, -1 } },//S5L5 
		{ {	-1,	-4, -6, -4,	-1	},	{ 2, 8, 12, 8, 2 }, { 0, 0, 0, 0, 0 }, { -2, -8, -12, -8, -2 }, { 1, 4, 6, 4, 1 } },//W5L5
		{ {	1,	4,	6,	4,	1	},		{ -4, -16, -24, -16, -4 }, { 6, 24, 36, 24, 6 }, { -4, -16, -24, -16, -4 }, { 1, 4, 6, 4, 1 } },//R5L5
		{ { 1,	2,	0,	-4,	-1	}, { 0, 0, 0, 0, 0 }, { -2, -4, 0, 8, 2 }, { 0, 0, 0, 0, 0 }, { 1, 2, 0, -4, -1 } },//S5E5
		{ { 1, 2, 0, -4, -1 }, { -2, -4, 0, 8, 2 }, { 0, 0, 0, 0, 0 }, { 2, 4, 0, -8, -2 }, { -1, -2, 0, 4, 1 } },//W5E5
		{ { -1, -2, 0, 4, 1 }, { 4, 8, 0, -16, -4 }, { -6, -12, 0, 24, 6 }, { 4, 8, 0, -16, -4 }, { -1, -2, 0, 4, 1 } },//R5E5
		{ { 1, 0, -2, 0, 1 }, { -2, 0, 4, 0, -2 }, { 0, 0, 0, 0, 0 }, { 2, 0, -4, 0, 2 }, { -1, 0, 2, 0, -1 } },//W5S5
		{ { -1, 0, 2, 0, -1 }, { 4, 0, -8, 0, 4 }, { -6, 0, 12, 0, -6 }, { 4, 0, -8, 0, 4 }, { -1, 0, 2, 0, -1 } },//R5S5
		{ { -1, 2, 0, -2, 1 }, { 4, -8, 0, 8, -4 }, { -6, 12, 0, -12, 6 }, { 4, -8, 0, 8, -4 }, { -1, 2, 0, -2, 1 } },//R5W5
		{ { 1, 2, 0, -4, -1 }, { 2, 4, 0, -8, -2 }, { 0, 0, 0, 0, 0 }, { -4, -8, 0, 16, 4 }, { -1, -2, 0, 4, 1 } },//E5E5
		{ { 1, 0, -2, 0, 1 }, { 0, 0, 0, 0, 0 }, { -2, 0, 4, 0, -2 }, { 0, 0, 0, 0, 0 }, { 1, 0, -2, 0, 1 } },//S5S5
		{ { 1, -2, 0, 2, -1 }, { -2, 4, 0, -4, 2 }, { 0, 0, 0, 0, 0 }, { 2, -4, 0, 4, -2 }, { -1, 2, 0, -2, 1 } },//W5W5
		{ { 1, -4, 6, -4, 1 }, { -4, 16, -24, 16, -4 }, { 6, -24, 36, -24, 6 }, { -4, 16, -24, 16, -4 }, { 1, -4, 6, -4, 1 } }//R5R5
		};
int kernels2[14][5][5] = {
		{ { -1, -2, 0, 4, 1 }, { -4, -8, 0, 16, 4 }, { -6, -12, 0, 24, 6 }, { -4, -8, 0, 16, 4 }, { -1, -2, 0, 4, 1 } }, //L5E5
		{ { -1, 0, 2, 0, -1 }, { -4, 0, 8, 0, -4 }, { -6, 0, 12, 0, -6 }, { -4, 0, 8, 0, -4 }, { -1, 0, 2, 0, -1 } },	//L5S5
		{ { -1, 2, 0, -2, 1 }, { -4, 8, 0, -8, 4 }, { -6, 12, 0, -12, 6 }, { -4, 8, 0, -8, 4 }, { -1, 2, 0, -2, 1 } },	//L5W5
		{ { 1, -4, 6, -4, 1 }, { 4, -16, 24, -16, 4 }, { 6, -24, 36, -24, 6 }, { 4, -16, 24, -16, 4 }, { 1, -4, 6, -4, 1 } },	//L5R5
		{ { 1, 0, -2, 0, 1 }, { 2, 0, -4, 0, 2 }, { 0, 0, 0, 0, 0 }, { -4, 0, 8, 0, -4 }, { -1, 0, 2, 0, -1 } },	//E5S5
		{ { 1, -2, 0, 2, -1 }, { 2, -4, 0, 4, -2 }, { 0, 0, 0, 0, 0 }, { -4, 8, 0, -8, 4 }, { -1, 2, 0, -2, 1 } },	//E5W5
		{ { -1, 4, -6, 4, -1 }, { -2, 8, -12, 8, -2 }, { 0, 0, 0, 0, 0 }, { 4, -16, 24, -16, 4 }, { 1, -4, 6, -4, 1 } },	//E5R5
		{ { 1, -2, 0, 2, -1 }, { 0, 0, 0, 0, 0 }, { -2, 4, 0, -4, 2 }, { 0, 0, 0, 0, 0 }, { 1, -2, 0, 2, -1 } },	//S5W5
		{ { -1, 4, -6, 4, -1 }, { 0, 0, 0, 0, 0 }, { 2, -8, 12, -8, 2 }, { 0, 0, 0, 0, 0 }, { -1, 4, -6, 4, -1 } },	//S5R5
		{ { -1, 4, -6, 4, -1 }, { 2, -8, 12, -8, 2 }, { 0, 0, 0, 0, 0 }, { -2, 8, -12, 8, -2 }, { 1, -4, 6, -4, 1 } },	//W5R5
		{ { 1, 2, 0, -4, -1 }, { 2, 4, 0, -8, -2 }, { 0, 0, 0, 0, 0 }, { -4, -8, 0, 16, 4 }, { -1, -2, 0, 4, 1 } },//E5E5
		{ { 1, 0, -2, 0, 1 }, { 0, 0, 0, 0, 0 }, { -2, 0, 4, 0, -2 }, { 0, 0, 0, 0, 0 }, { 1, 0, -2, 0, 1 } },//S5S5
		{ { 1, -2, 0, 2, -1 }, { -2, 4, 0, -4, 2 }, { 0, 0, 0, 0, 0 }, { 2, -4, 0, 4, -2 }, { -1, 2, 0, -2, 1 } },//W5W5
		{ { 1, -4, 6, -4, 1 }, { -4, 16, -24, 16, -4 }, { 6, -24, 36, -24, 6 }, { -4, 16, -24, 16, -4 }, { 1, -4, 6, -4, 1 } }//R5R5
};

#define SOBEL_DERIVATIVE

double * edge_sharpness(int *derivative_Rx, int *derivative_Gx, int *derivative_Bx,
						int *derivative_Ry, int *derivative_Gy,
						int *derivative_By, IplImage *edge_Image);

int checkneighbors(unsigned char * canny, unsigned char * weakEdges, unsigned char *visited, int i, int j, int imageW, int imageH);
int myCanny(IplImage *src, unsigned char * canny_image);
int convolution2d(IplImage *padded_32b,int i,int j,int laws_num);
int connectgaps(unsigned char * img, int imageW, int imageH);

int max(int a, int b) {
	if (a > b) 
		return a;
	else
		return b;
}

int min(int a, int b) {
	if (a < b)
		return a;
	else
		return b;
}

int  distance_maxpoint(g *image_g, int y, int x, int imageH, int imageW, int direction,int limit){

	/*
	FUNCTION INFO:------distance_maxpoint return the relative distance of the the maximum point around x-limit,y-limit with the x,y point
						it can search either horizontally(direction == 1) or vertically (direction == 0). When we search horizontally we
						check the y-magnitude, when we check vertically we check the x-magnitude
	INPUT:--------------buffer:	the magnitude buffer type g
						x,y:	the point
						direction:the direction
						limit:	the limit of the search
	OUPUT:--------------return value is the realtive distance of the max with x,y
	*/

	int i;
	int max_coor;
	int distancefrompoint;
	
	if (direction == 0) { 
		max_coor = y;
		for (i = max(y - limit, 0); i < min(y + limit, imageH); i++) {
			if (image_g[i*imageW + x].magn_x>image_g[max_coor*imageW + x].magn_x) {
				max_coor = i;
			}
		}
		//distancefrompoint = max_coor - limit + (y<limit)*(limit - y);
		distancefrompoint = max_coor - y;
	}
	if (direction == 1) {
		max_coor = x;
		for (i = max(x - limit, 0); i < min(x + limit, imageW); i++) {
			if (image_g[y*imageW + i].magn_y>image_g[y*imageW + max_coor].magn_y) {
				max_coor = i;
			}
		}
		//distancefrompoint = max_coor - limit + (x<limit)*(limit - x);
		distancefrompoint = max_coor - x;
	}
	return distancefrompoint;
} 

double * edge_sharpness(int *derivative_Rx,int *derivative_Gx,int *derivative_Bx,
					   int *derivative_Ry,int *derivative_Gy,
					   int *derivative_By,IplImage *edge_Image) {
	
	/*
	FUNCTION INFO:------edge_sharpness calculates siga, a value for measuring the sharpness of an edge
	INPUT:--------------derivative_ChDir:	buffers with the sobel derivatives of the image
						edge_Image:			Iplbuffer with 0 and 255 values withe the canny detector results
	OUPUT:--------------sigma:				double buffer with the sigma values
	MATLAB DIFF:--------The diffrences with Matlab are at 0.01 i believe that the errors exist for 2 reasons, the 
						FP operations, and some diffrences in the canny points that come from FP operations too
						Also there are not implemented averaging techniques.
	*/


	int rx,gx,bx,ry,gy,by;
	int gxx, gyy, gxy;
	double gtheta_a,gtheta_b,theta;
	struct g *image_g;
	int i,j;

	int imageW=edge_Image->width;
	int imageH=edge_Image->height;
	int widthStep=edge_Image->widthStep;

	image_g=(struct g *)malloc(imageW*imageH*sizeof(struct g));

	for (i=0;i<imageH;i++) {
		for (j=0;j<imageW;j++) {

			rx = derivative_Rx[i*imageW + j];
			gx = derivative_Gx[i*imageW + j];
			bx = derivative_Bx[i*imageW + j];
			ry = derivative_Ry[i*imageW + j];
			gy = derivative_Gy[i*imageW + j];
			by = derivative_By[i*imageW + j];
			gxx = rx*rx + gx*gx + bx*bx;
			gyy = ry*ry + gy*gy + by*by;
			gxy = rx*ry + gx*gy + bx*by;

			theta = 0.5 * (atan2(2*gxy, gxx - gyy));
			
			gtheta_a = 0.5 * ((gxx + gyy) + (gxx - gyy)*
				cos(2 * theta) + 2 * gxy * sin(2 * theta));

			gtheta_b = 0.5 * ((gxx + gyy) + (gxx - gyy)*
				cos(2 * (theta + PI / 2)) + 2 * gxy*sin(2 * (theta + PI / 2)));

			gtheta_a = sqrt(abs(gtheta_a));
			gtheta_b = sqrt(abs(gtheta_b));

			if (gtheta_a > gtheta_b) {
				image_g[i*imageW + j].gtheta = gtheta_a;
				image_g[i*imageW + j].magn_x = abs(gtheta_a * cos(theta));
				image_g[i*imageW + j].magn_y = abs(gtheta_a * sin(theta));
				image_g[i*imageW + j].theta = theta;
			}
			else {
				image_g[i*imageW + j].gtheta = gtheta_b;
				image_g[i*imageW + j].magn_x = abs((gtheta_b * cos(theta + PI / 2)));
				image_g[i*imageW + j].magn_y = abs((gtheta_b * sin(theta + PI / 2)));
				image_g[i*imageW + j].theta = (theta + PI / 2);
			}
		}
	}
	
	//find	max around canny point

	int *distancefromcanny_x = (int *)calloc(imageW*imageH,sizeof(int));
	int *distancefromcanny_y = (int *)calloc(imageW*imageH,sizeof(int));

	for (i = 0; i < imageH; i++) {//need to doublecheck the correctness of the calculations below
		for (j = 0; j < imageW; j++) {
			if ((unsigned char)edge_Image->imageData[i*widthStep + j] == 255) {
				distancefromcanny_x[i*imageW + j] = distance_maxpoint(image_g, i, j, imageH, imageW, 0, DISTANCE_FROM_CANNY);
				distancefromcanny_y[i*imageW + j] = distance_maxpoint(image_g, i, j, imageH, imageW, 1, DISTANCE_FROM_CANNY);
				if (abs(distancefromcanny_x[i*imageW + j]) + abs(distancefromcanny_y[i*imageW + j]) > 2 * DISTANCE_FROM_CANNY) {
					edge_Image->imageData[i*widthStep + j] = 0; //we assume its a canny error											
				}
			}
		}
	}


	/*antistoixia metavlitwn matlab me c
	imgradYdis      -> distancefromcanny_y
	imgradXdis      -> distancefromcanny_x
	imgradY         -> image_g[].magn_y
	imgradX         -> image_g[].magn_x
	sigmaX,sigmaY   -> sigma
	p				-> offset
	*/
	int i1,j1,offset;
	double gg,magn_x,magn_y,sum,ssigma,curvelength,temp_pow;
	double *image_s = (double *)calloc(imageW*imageH, sizeof(double));
	for (i = 0; i < imageH; i++) {
		for (j = 0; j < imageW; j++) {
			if ((unsigned char)edge_Image->imageData[i*widthStep + j] == 255) {
				if (image_g[i*imageW + j].magn_y >= image_g[i*imageW + j].magn_x) {
					offset = 0;
					j1 = j + distancefromcanny_y[i*imageW + j];
					magn_y = image_g[i*imageW + j1].magn_y;
					magn_x = image_g[i*imageW + j1].magn_x;
					gg = magn_y / sqrt(magn_x*magn_x + magn_y*magn_y); //what is magn_xx and magn_y = 0;
					sum = magn_y;
					ssigma = 0;
					curvelength = 0;
					while (j1 + offset - 1 >= 0) {//
						if ((image_g[i*imageW + j1 + offset - 1].magn_y > 0.1*magn_y) 
							&&(image_g[i*imageW + j1 + offset].magn_y - image_g[i*imageW + j1 + offset - 1].magn_y)
							> 0.05*magn_y) {

							temp_pow = pow((image_g[i*imageW + j1 + offset - 1].magn_y / magn_y) - (image_g[i*imageW + j1 + offset].magn_y / magn_y), 2);
							curvelength += sqrt(temp_pow + 1);
							ssigma += image_g[i*imageW + j1 + offset - 1].magn_y * curvelength *curvelength;
							sum += image_g[i*imageW + j1 + offset - 1].magn_y;
							offset--;
						}
						else {
							break;
						}
					}
					offset = 0;
					curvelength = 0;
					while (j1 + offset + 1 < imageW) {
						if ((image_g[i*imageW + j1 + offset + 1].magn_y > 0.1*magn_y) &&
							(image_g[i*imageW + j1 + offset].magn_y - image_g[i*imageW + j1 + offset + 1].magn_y) > 0.05*magn_y) {

							temp_pow = pow((image_g[i*imageW + j1 + offset + 1].magn_y / magn_y) - (image_g[i*imageW + j1 + offset].magn_y / magn_y), 2);
							curvelength += sqrt(temp_pow + 1);
							ssigma += image_g[i*imageW + j1 + offset + 1].magn_y * curvelength *curvelength;
							sum += image_g[i*imageW + j1 + offset + 1].magn_y;
							offset++;
						}
						else {
							break;
						}
					}
					ssigma = sqrt(ssigma / sum);
					image_s[i*imageW + j] = ssigma*gg;
				}
				if (image_g[i*imageW + j].magn_y < image_g[i*imageW + j].magn_x) {
					offset = 0;
					i1 = i + distancefromcanny_x[i*imageW + j];
					magn_x = image_g[i1*imageW + j].magn_x;
					magn_y = image_g[i1*imageW + j].magn_y; 
					gg = magn_x / (magn_y*magn_y + magn_x*magn_x); //gy and magn_x = 0 ?
					sum = magn_x;
					curvelength = 0;
					ssigma = 0;
					while (i1 + offset - 1 >= 0) {
						if ((image_g[(i1 + offset - 1)*imageW + j].magn_x > 0.1*magn_x) &&
							((image_g[(i1 + offset)*imageW + j].magn_x - image_g[(i1 + offset - 1)*imageW + j].magn_x) > 0.05*magn_x)) {

							temp_pow = pow(image_g[(i1 + offset - 1)*imageW + j].magn_x / magn_x - image_g[(i1 + offset)*imageW + j].magn_x / magn_x, 2);
							curvelength += sqrt(temp_pow + 1);
							ssigma += image_g[i1 + offset - 1].magn_x * curvelength*curvelength;
							sum += image_g[i1 + offset - 1].magn_x;
							offset--;
						}
						else {
							break;
						}
					}
					offset = 0;
					curvelength = 0;
					while (i1 + offset + 1 < imageH) {
						if ((image_g[(i1 + offset + 1)*imageW + j].magn_x > 0.1*magn_x) &&
							((image_g[(i1 + offset)*imageW + j].magn_x - image_g[(i1 + offset + 1)*imageW + j].magn_x) > 0.05*magn_x)) {

							temp_pow = pow(image_g[(i1 + offset + 1)*imageW + j].magn_x / magn_x - image_g[(i1 + offset)*imageW + j].magn_x / magn_x, 2);
							curvelength += sqrt(temp_pow + 1);
							ssigma += image_g[i1 + offset + 1].magn_x * curvelength*curvelength;
							sum += image_g[i1 + offset + 1].magn_x;
							offset++;
						}
						else {
							break;
						}
					}
					ssigma = sqrt(ssigma / sum);
					image_s[i*imageW + j] = ssigma * gg;
				}
			}
		}
	}


	return image_s;
}

int * manual_thresholding_sigma(double *sigma, int imageW, int imageH, double thr) {
	int i, j;
	double temp_sigma;
	int *sigma_thr = (int *)calloc(imageW*imageH, sizeof(int));
	for (i = 0; i < imageH; i++) {
		for (j = 0; j < imageW; j++ ) {
			temp_sigma = sigma[i*imageW + j];
			if (temp_sigma != 0) {
				if (abs(temp_sigma)>thr) {
					sigma_thr[i*imageW + j] = 2;
				}
				else {
					sigma_thr[i*imageW + j] = 3;
				}
			}
		}
	}
	return sigma_thr;
}

float convolution_laws(IplImage *padded_32b,int posy,int posx,int laws_num) {

	int i,j;
	float res1,res2;
	res1 = 0.0;
	res2 = 0.0;
	//int widthStep=padded_32b->widthStep;

	for (i = -2; i <= 2; i++) {
		for (j = -2; j <= 2; j++) {
			res1 += CV_IMAGE_ELEM(padded_32b, float, posy + 2 + i, posx + 2 + j)*kernels1[laws_num][i + 2][j + 2];
			res2 += CV_IMAGE_ELEM(padded_32b, float, posy + 2 + i, posx + 2 + j)*kernels2[laws_num][i + 2][j + 2];
		}
	}

	return res1 + res2;
}

int getSobelvalue_x(IplImage *buffer, int k, int t) {

	/*
	FUNCTION INFO:-----getSobelvalue_y return the value of a vertical sobel filter at the k,t coordinates
	INPUT:--------------buffer:	the image buffer where the convolution will be applied
	OUPUT:--------------return value
	*/

	int i, j;
	int res;
	res = 0;
	for (i = -1; i <= 1; i++){
		for (j = -1; j <= 1; j++) {
			res += CV_IMAGE_ELEM(buffer,unsigned char,k+i,j+t)*sobel_x[i + 1][j + 1];
		}
	}
	return res;
}

int getSobelvalue_y(IplImage *buffer, int k, int t) {
	/*
	FUNCTION INFO:-----getSobelvalue_y return the value of a vertical sobel filter at the k,t coordinates 
	INPUT:--------------buffer:	the image buffer where the convolution will be applied
	OUPUT:--------------return value
	*/
	int i, j;
	int res;
	res = 0;
	for (i = -1; i <= 1; i++){
		for (j = -1; j <= 1; j++) {
			res += CV_IMAGE_ELEM(buffer, unsigned char, k + i, j + t)*sobel_y[i + 1][j + 1];
		}
	}
	return res;
}

double convolution2d(IplImage *buffer, int posy, int posx, double *mask) {
	
	/*
	FUNCTION INFO:-----convolution2d convolves a mask/kernel with an image at a specific point.
	INPUT:--------------buffer:		IplImage pointer of the image that has depth, either IPL_DEPTH_64F or IPL_DEPTH_8U 
						posx,posy:	The potition that convolution will be computed, THE CALLER MUST BE SURE that there are no out of bounds exceptions
						mask:		The mask/kernel must be 7x7
	OUPUT:--------------dst: A pointer to double that has to filtered version of img
	MATLAB DIFFRENCES:--There are diffrence +-0.001 that i believe come from the order of the DP prakseis
	TODO:---------------Parameterize for diffrent kernel sizes
	*/
	
	int i, j;
	double res;

	res = 0;

	if (buffer->depth==IPL_DEPTH_8U) {
		for (i = -3; i <= 3; i++) {
			for (j = -3; j <= 3; j++) {
				res += CV_IMAGE_ELEM(buffer, unsigned char, posy + i, posx + j)* mask[(i + 3) * 7 + j + 3];
			}
		}
	}

	if (buffer->depth==IPL_DEPTH_64F) {
		for (i = -3; i <= 3; i++) {
			for (j = -3; j <= 3; j++) {
				res += CV_IMAGE_ELEM(buffer, double, posy + i, posx + j)* mask[(i + 3) * 7 + j + 3];
			}
		}
	}


	return (res);
}

void filter_DP_2d(IplImage *img, double *dst, double *kernel) {
	/*
	FUNCTION INFO:-----filter_F_2d takes an IplImage pointer and filters all the image with a 2d float 8x8 kernel taking a double result
	INPUT:--------------img:	IplImage pointer that has a non-bordered, one channel image. It has to have depth IPL_DEPTH_64F or IPL_DEPTH_8U
						kernel: double pointer that has an exactly 7x7 kernel, if the kernel is larget we take the middle part, 
								if it is smaller it crashes
	OUPUT:--------------dst: A pointer to double that has to filtered version of img 
	MATLAB DIFFRENCES:--There are diffrence +-0.001 that i believe come from the order of the DP prakseis
	TODO:---------------Parameterize for diffrent kernel sizes
	*/

	int i, j;
	int imageW = img->width;
	int imageH = img->height;
	IplImage *img_border;
	

	int type = img->depth;
	img_border = cvCreateImage(cvSize(img->width + 6, img->height + 6), type, 1);
	
	CvPoint b;
	b.x = 3;
	b.y = 3;
	cvCopyMakeBorder(img, img_border, b, IPL_BORDER_REPLICATE);

	for (i = 0; i < imageH; i++){
		for (j = 0; j < imageW; j++) {
			dst[i*imageW + j] = convolution2d(img_border, i + 3, j + 3, kernel);
		}
	}
}

int myColorCanny(IplImage *src, unsigned char * canny_image) {

	/*
	FUNCTION INFO:------myColorCanny function is a canny implementation that takes into account
	the color information of an image.
	INPUT:--------------An IplImage pointer with an RGB image
	OUPUT:--------------A binary image that gives the canny points
	MATLAB DIFFRENCES:--I believe the only diffrences i have with the matlab code is because
						of the FP operations (they are done in diffrent order). In the magnitude
						the fault is +-0.0001, but it gives some diffrent canny points.
	TODO:---------------Parameterize the Gaussian so the "user-programmer" can try diffrent size of gaussian masks
						and diffrent sigmas.
	*/

	IplImage *src_gau_R, *src_gau_G, *src_gau_B, *src_gau;
	int i, j;

	src_gau = cvCreateImage(cvSize(src->width, src->height), 8, 3);
	src_gau_R = cvCreateImage(cvSize(src->width, src->height), 8, 1);
	src_gau_G = cvCreateImage(cvSize(src->width, src->height), 8, 1);
	src_gau_B = cvCreateImage(cvSize(src->width, src->height), 8, 1);
	int imageW = src->width;
	int imageH = src->height;

	//split the channels
	cvSplit(src, src_gau_B, src_gau_G, src_gau_R, NULL);

	//create gaussian filter 2-d 7x7
	double *gau = (double *)malloc(7 * 7 * sizeof(double));

	double sig = 0.7;
	double tmp;
	for (i = -3; i < 4; i++) {
		for (j = -3; j <= 3; j++) {
			tmp = exp((double)(-i*i - j*j) / (2 * sig*sig));
			gau[(i + 3) * 7 + j + 3] = tmp / ((2 * PI*sig*sig)*(2 * PI*sig*sig));
		}
	}

	//filter RGB channels with the gaussian
	double * R_blurred = (double *)malloc(imageW*imageH*sizeof(double));
	double * G_blurred = (double *)malloc(imageW*imageH*sizeof(double));
	double * B_blurred = (double *)malloc(imageW*imageH*sizeof(double));

	filter_DP_2d(src_gau_R, R_blurred, gau);
	filter_DP_2d(src_gau_G, G_blurred, gau);
	filter_DP_2d(src_gau_B, B_blurred, gau);

	//pass the result of the filtering to IplImage buffers
	IplImage *R_blurred_double = cvCreateImage(cvSize(imageW, imageH), IPL_DEPTH_64F, 1);
	IplImage *G_blurred_double = cvCreateImage(cvSize(imageW, imageH), IPL_DEPTH_64F, 1);
	IplImage *B_blurred_double = cvCreateImage(cvSize(imageW, imageH), IPL_DEPTH_64F, 1);

	for (i = 0; i < imageH; i++){
		for (j = 0; j < imageW; j++) {
			CV_IMAGE_ELEM(R_blurred_double, double, i, j) = R_blurred[i*imageW + j];
			CV_IMAGE_ELEM(G_blurred_double, double, i, j) = G_blurred[i*imageW + j];
			CV_IMAGE_ELEM(B_blurred_double, double, i, j) = B_blurred[i*imageW + j];
		}
	}

	//Compute diffrence of gaussians 7x7 for y-axis
	for (i = -3; i <= 3; i++) {
		for (j = -3; j <= 3; j++) {
			tmp = exp((double)(-i*i - j*j) / (2 * sig*sig));
			gau[(i + 3) * 7 + j + 3] = -i * tmp / (PI*sig*sig);
		}
	}

	//filter the previous blurred channels with diffrence of gaussian y-axis
	double * Ry = (double *)malloc(imageW*imageH*sizeof(double));
	double * Gy = (double *)malloc(imageW*imageH*sizeof(double));
	double * By = (double *)malloc(imageW*imageH*sizeof(double));
	

	filter_DP_2d(R_blurred_double, Ry, gau);
	filter_DP_2d(G_blurred_double, Gy, gau);
	filter_DP_2d(B_blurred_double, By, gau);


	//Compute diffrence of gaussians 7x7 for x-axis
	for (i = -3; i <= 3; i++) {
		for (j = -3; j <= 3; j++) {
			tmp = exp((double)(-j*j - i*i) / (2 * sig*sig));
			gau[(i + 3) * 7 + j + 3] = -j * tmp / (PI*sig*sig);
		}
	}
	//filter the previous blurred channels with diffrence of gaussian x-axis
	double * Rx = (double *)malloc(imageW*imageH*sizeof(double));
	double * Gx = (double *)malloc(imageW*imageH*sizeof(double));
	double * Bx = (double *)malloc(imageW*imageH*sizeof(double));

	filter_DP_2d(R_blurred_double, Rx, gau);
	filter_DP_2d(G_blurred_double, Gx, gau);
	filter_DP_2d(B_blurred_double, Bx, gau);


	//compute the magnitude of the color gradients
	struct g * image_g = (struct g *)malloc(imageW*imageH*sizeof(struct g));
	double rx, gx, bx, ry, gy, by;
	double gxx, gyy, gxy, theta, gtheta_a, gtheta_b;



	for (i = 0; i < imageH; i++) {
		for (j = 0; j < imageW; j++) {
			rx = -Rx[i*imageW + j];
			gx = -Gx[i*imageW + j];
			bx = -Bx[i*imageW + j];
			ry = -Ry[i*imageW + j];
			gy = -Gy[i*imageW + j];
			by = -By[i*imageW + j];
			gxx = (rx*rx + gx*gx + bx*bx);
			gyy = (ry*ry + gy*gy + by*by);
			gxy = (rx*ry + gx*gy + bx*by);


			theta = 0.5 * atan2(2*gxy, gxx - gyy);		


			gtheta_a = 0.5 * ((gxx + gyy) + (gxx - gyy)*
				cos(2 * theta) + 2 * gxy * sin(2 * theta));

			gtheta_b = 0.5 * ((gxx + gyy) + (gxx - gyy)*
				cos(2 * (theta + PI / 2)) + 2 * gxy*sin(2 * (theta + PI / 2)));

			gtheta_a = sqrt(abs(gtheta_a));
			gtheta_b = sqrt(abs(gtheta_b));

			if (gtheta_a > gtheta_b) {
				image_g[i*imageW + j].gtheta = gtheta_a;
				image_g[i*imageW + j].magn_x = abs(gtheta_a * cos(theta));
				image_g[i*imageW + j].magn_y = abs(gtheta_a * sin(theta));
				image_g[i*imageW + j].theta = theta;
			}
			else {
				image_g[i*imageW + j].gtheta = gtheta_b;
				image_g[i*imageW + j].magn_x = abs((gtheta_b * cos(theta + PI / 2)));
				image_g[i*imageW + j].magn_y = abs((gtheta_b * sin(theta + PI / 2)));
				image_g[i*imageW + j].theta = (theta + PI / 2);
			}
		}
	}

	//find max min
	double gmax = 0;
	double gmin = 100000.0;
	for (i = 0; i < imageH; i++) {
		for (j = 0; j < imageW; j++) {
			if (image_g[i*imageW + j].gtheta>gmax) {
				gmax = image_g[i*imageW + j].gtheta;
			}
			if (image_g[i*imageW + j].gtheta<gmin) {
				gmin = image_g[i*imageW + j].gtheta;
			}
		}
	}

	//Normalize gtheta 0 - 1
	for (i = 0; i < imageH; i++){
		for (j = 0; j < imageW; j++){
			image_g[i*imageW + j].gtheta = (image_g[i*imageW + j].gtheta - gmin) / (gmax - gmin);
		}
	}

	//Find local maximums and decide if they are weak or strong edges 
	double ay, ax, d, grad1, grad2;
	int flag;
	unsigned char * localMax = (unsigned char *)calloc(imageW*imageH, sizeof(unsigned char));
	unsigned char * weakEdges = (unsigned char *)calloc(imageW*imageH, sizeof(unsigned char));
	unsigned char * strongEdges = (unsigned char *)calloc(imageW*imageH, sizeof(unsigned char));
	for (i = 1; i < imageH - 1; i++) {
		for (j = 1; j < imageW - 1; j++)	{
			ax = image_g[i*imageW + j].magn_x;
			ay = image_g[i*imageW + j].magn_y;
			flag = 0;
			//check the direction that the edge has (0-45 45-90 90-135 135-180) and then calculate the linear interpolation
			if ((ay <= 0 && ax > -ay) || (ay >= 0 && ax < -ay)) {
				d = abs(ay / ax);
				grad1 = image_g[i*imageW + j + 1].gtheta*(1 - d) +
					image_g[(i - 1)*imageW + j + 1].gtheta*d;
				grad2 = image_g[i*imageW + j - 1].gtheta*(1 - d) +
					image_g[(i + 1)*imageW + j - 1].gtheta*d;
				flag = 1;
			}
			else if ((ax > 0 && -ay >= ax) || (ax<0 && -ay <= ax)) {
				d = abs(ax / ay);
				grad1 = image_g[(i - 1)*imageW + j].gtheta*(1 - d) +
					image_g[(i - 1)*imageW + j + 1].gtheta*d;
				grad2 = image_g[(i + 1)*imageW + j].gtheta*(1 - d) +
					image_g[(i + 1)*imageW + j - 1].gtheta*d;
				flag = 1;
			}
			else if ((ax <= 0 && ax>ay) || (ax >= 0 && ax < ay)){
				d = abs(ax / ay);
				grad1 = image_g[(i - 1)*imageW + j].gtheta*(1 - d) +
					image_g[(i - 1)*imageW + j - 1].gtheta*d;
				grad2 = image_g[(i + 1)*imageW + j].gtheta*(1 - d) +
					image_g[(i + 1)*imageW + j + 1].gtheta*d;
				flag = 1;
			}
			else if ((ay < 0 && ax <= ay) || (ay>0 && ax >= ay)) {
				d = abs(ay / ax);
				grad1 = image_g[i*imageW + j - 1].gtheta*(1 - d) +
					image_g[(i - 1)*imageW + j - 1].gtheta*d;
				grad2 = image_g[i*imageW + j + 1].gtheta*(1 - d) +
					image_g[(i + 1)*imageW + j + 1].gtheta*d;
				flag = 1;
			}
			//if the center pixel is larger than both other its local maximum
			if (image_g[i*imageW + j].gtheta >= grad1 && image_g[i*imageW + j].gtheta >= grad2 && flag == 1) {
				localMax[i*imageW + j] = 1;
				if (image_g[i*imageW + j].gtheta > CANNY_THRESHOLD_1) {
					weakEdges[i*imageW + j] = 1;
				}
				if (image_g[i*imageW + j].gtheta > CANNY_THRESHOLD_2) {
					strongEdges[i*imageW + j] = 1;
				}
			}
		}
	}


	//make weakedges and strongedges 0 in borders so we dont go out of borders when we check for neighbors
	for (i = 0; i < imageH; i++) {
		for (j = 0; j < imageW; j++) {
			if (i == 0 || j == 0 || i == imageH - 1 || j == imageW - 1){
				weakEdges[i*imageW + j] = 0;
				strongEdges[i*imageW + j] = 0;
			}
		}
	}

	//edges are all the weak edges that are 8-connected with a strong edge
	unsigned char * visited = (unsigned char *)calloc(imageW*imageH, sizeof(unsigned char));
	for (i = 0; i < imageH; i++){
		for (j = 0; j < imageW; j++) {
			if (strongEdges[i*imageW + j] == 1) {
				visited[i*imageW + j] = 1;
				canny_image[i*imageW + j] = 1;
				checkneighbors(canny_image, weakEdges, visited, i, j, imageW, imageH);
			}
		}
	}

	//connect 1-pixel gaps
	connectgaps(canny_image, imageW, imageH);

	//remove the cutted off edges (alone)
	int k, w;
	for (i = 1; i < imageH - 1; i++){
		for (j = 1; j < imageW - 1; j++) {
			flag = 0;
			for (k = -1; k <= 1; k++) {
				for (w = -1; w <= 1; w++){
					if (canny_image[(i + k)*imageW + j + w] == 1) {
						flag = 1;
					}
				}
			}
			if (flag == 0) {
				canny_image[i*imageW + j] = 0;
			}
		}
	}

	/*// FOR DEBUGGING PURPOSES WE COUNT SOME STUFF
	int countstrong = 0;
	int countweak = 0;
	int countmax = 0;
	for (i = 0; i < imageH; i++) {
	for (j = 0; j < imageW; j++) {
	if (strongEdges[i*imageW + j] == 1) {
	countstrong++;
	}
	if (weakEdges[i*imageW + j] == 1) {
	countweak++;
	}
	if (localMax[i*imageW + j] == 1) {
	countmax++;
	}
	}
	}*/

	cvReleaseImage(&src_gau_R);
	cvReleaseImage(&src_gau_G);
	cvReleaseImage(&src_gau_B);
	cvReleaseImage(&src_gau);
	cvReleaseImage(&R_blurred_double);
	cvReleaseImage(&G_blurred_double);
	cvReleaseImage(&B_blurred_double);
	free(R_blurred);
	free(G_blurred);
	free(B_blurred);
	free(localMax);
	free(weakEdges);
	free(strongEdges);
	free(Rx);
	free(Ry);
	free(Gx);
	free(Gy);
	free(Bx);
	free(By);
	free(visited);

	return 0;
}

int connectgaps(unsigned char * img, int imageW, int imageH) {

	/*
	FUNCTION INFO:------Connectedgaps function connects 1-pixels gaps in a binary image the function is in-place operation.
	INPUT:--------------An unsigned char pointer to a buffer that has only binary values.
	OUPUT:--------------Is the same buffer that has the inputs modified in-place.
	MATLAB DIFFRENCES:--Its the exact same implementation, i believe no error is occured in this function
	TODO:---------------Nothing
	*/

	int i, j;
	int num_id;
	for (i = 1; i < imageH - 1; i++) {
		for (j = 1; j < imageW - 1; j++) {
			if (img[i*imageW + j] == 0) {
				num_id = img[(i - 1)*imageW + j - 1] * 1 + img[(i - 1)*imageW + j] * 2 +
					img[(i - 1)*imageW + j + 1] * 4 + img[i*imageW + j + 1] * 8 +
					img[(i + 1)*imageW + j + 1] * 16 + img[(i + 1) * imageW + j] * 32 +
					img[(i + 1)*imageW + j - 1] * 64 + img[i*imageW + j - 1] * 128;
			}
			if (num_id == 32 + 2 || num_id == 1 + 4 || num_id == 128 + 8 || num_id == 16 + 64 ||
				num_id == 2 + 128 || num_id == 128 + 4 || num_id == 2 + 8 || num_id == 1 + 8 ||
				num_id == 64 + 8 || num_id == 8 + 32 || num_id == 4 + 32 || num_id == 1 + 32 ||
				num_id == 1 + 16 || num_id == 1 + 64 || num_id == 2 + 64 || num_id == 2 + 16 ||
				num_id == 4 + 16 || num_id == 32 + 128) {
				img[i*imageW + j] = 1;
			}

		}
	}
	return 1;
}

int checkneighbors(unsigned char * out_buffer, unsigned char * weakEdges, unsigned char *visited, int i, int j, int imageW, int imageH) {

	/*
	FUNCTION INFO:------Check neighbors calculates 8-connected objects that start in a specific point and puts 1
	in the output buffer where the object is.
	INPUT:--------------weakEdges:	pointer to unsigned char, buffer with the binary picture, MUST HAVE 0 TO THE BORDER!
						visited:	pointer to unsigned char, buffer that has 1 in place that we have visited or we want to ignore
						i,j:		the coordinates from a point that our object has.
						imageW/H:	the size of the weakEdges-visited buffer, they are obligated on the same size
	OUPUT:--------------out_buffer:	pointer to unsigned char, buffer that gets 1 where the object is, CAREFULL if you want to have just
						the one object it has to be initialized to zero before the function is called.
	MATLAB DIFFRENCES:--None that come to my observation.
	TODO:---------------Nothing.
	*/

	int k, w;

	for (k = -1; k <= 1; k++) {
		for (w = -1; w <= 1; w++) {
			if (k != 0 || w != 0) {
				if (weakEdges[(i + k) + j + w] == 1 && visited[(i + k)*imageW + j + w] == 0) {
					visited[(i + k)*imageW + j + w] = 1;
					out_buffer[(i + k)*imageW + j + w] = 1;
					checkneighbors(out_buffer, weakEdges, visited, i + k, j + w, imageW, imageH);
				}
			}
		}
	}

	return 1;
}

int main(int argc, char *argv[]) {

	IplImage *original_Image,*original_Image_bw,*edge_Image,*gaussian_image_bw;
	IplImage *derivative_Rx,*derivative_Gx,*derivative_Bx;
	IplImage *derivative_Ry,*derivative_Gy,*derivative_By;
	IplImage *original_R,*original_B,*original_G;
	int i,j,k;
	int imageW,imageH;

	original_Image  = cvLoadImage(FILENAME,1); 
	original_Image_bw  = cvLoadImage(FILENAME,0);

	imageW=original_Image->width;
	imageH=original_Image->height;

	gaussian_image_bw = cvCreateImage(cvSize (original_Image->width, original_Image->height), 8, 1); 
	edge_Image = cvCreateImage (cvSize (original_Image->width, original_Image->height), 8, 1); 
	original_R = cvCreateImage (cvSize (original_Image->width, original_Image->height), 8, 1);
	original_G = cvCreateImage (cvSize (original_Image->width, original_Image->height), 8, 1);
	original_B = cvCreateImage (cvSize (original_Image->width, original_Image->height), 8, 1);
	derivative_Rx = cvCreateImage (cvSize (original_Image->width, original_Image->height), 8, 1);
	derivative_Gx = cvCreateImage (cvSize (original_Image->width, original_Image->height), 8, 1);
	derivative_Bx = cvCreateImage (cvSize (original_Image->width, original_Image->height), 8, 1);
	derivative_Ry = cvCreateImage (cvSize (original_Image->width, original_Image->height), 8, 1);
	derivative_Gy = cvCreateImage (cvSize (original_Image->width, original_Image->height), 8, 1);
	derivative_By = cvCreateImage (cvSize (original_Image->width, original_Image->height), 8, 1);


	int widthStep = derivative_Rx->widthStep;

	//smoothing before canny, matlab does it automatically, i was using this code before i wrote mycolorcanny
		/*cvSmooth(original_Image_bw,gaussian_image_bw,CV_GAUSSIAN,7,7,3);
		cvCanny(gaussian_image_bw,edge_Image,CANNY_THRESHOLD_1,CANNY_THRESHOLD_2,3); */

	unsigned char * canny_im = (unsigned char *)calloc(imageW*imageH,sizeof(unsigned char));
	myColorCanny(original_Image, canny_im);

	for (i = 0; i < imageH; i++) {
		for (j = 0; j < imageW; j++) {
			if (canny_im[i*imageW + j] == 1) {
				edge_Image->imageData[i*edge_Image->widthStep + j] = -1;
			}
			else {
				edge_Image->imageData[i*edge_Image->widthStep + j] = 0;
			}
		}
	}
	cvSaveImage("mycanny.jpg",edge_Image);
	cvSplit(original_Image,original_B,original_G,original_R,NULL);

#ifdef SOBEL_DERIVATIVE
	int *Rx, *Ry, *Gx, *Gy, *Bx, *By;
	IplImage *R_bordered, *G_bordered, *B_bordered;
	CvPoint b;
	b.x = 1;
	b.y = 1;
	R_bordered = cvCreateImage(cvSize(imageW + 2, imageH + 2), IPL_DEPTH_8U, 1);
	G_bordered = cvCreateImage(cvSize(imageW + 2, imageH + 2), IPL_DEPTH_8U, 1);
	B_bordered = cvCreateImage(cvSize(imageW + 2, imageH + 2), IPL_DEPTH_8U, 1);
	Rx = (int *)malloc(imageW*imageH*sizeof(int));
	Ry = (int *)malloc(imageW*imageH*sizeof(int));
	Gx = (int *)malloc(imageW*imageH*sizeof(int));
	Gy = (int *)malloc(imageW*imageH*sizeof(int));
	Bx = (int *)malloc(imageW*imageH*sizeof(int));
	By = (int *)malloc(imageW*imageH*sizeof(int));
	cvCopyMakeBorder(original_R,R_bordered,b,IPL_BORDER_REPLICATE);
	cvCopyMakeBorder(original_G,G_bordered,b,IPL_BORDER_REPLICATE);
	cvCopyMakeBorder(original_B,B_bordered,b,IPL_BORDER_REPLICATE);

	for (i=0;i<imageH;i++) {
		for(j=0;j<imageW;j++) {
			Rx[i*imageW + j] = getSobelvalue_x(R_bordered, i + 1, j + 1);
			Gx[i*imageW + j] = getSobelvalue_x(G_bordered, i + 1, j + 1);
			Bx[i*imageW + j] = getSobelvalue_x(B_bordered, i + 1, j + 1);
			Ry[i*imageW + j] = getSobelvalue_y(R_bordered, i + 1, j + 1);
			Gy[i*imageW + j] = getSobelvalue_y(G_bordered, i + 1, j + 1);
			By[i*imageW + j] = getSobelvalue_y(B_bordered, i + 1, j + 1);
		}
	}

#endif 

	//CALCULATE SIMGA
	double *sigma;
	sigma=edge_sharpness(Rx,Gx,Bx,Ry,Gy,By,edge_Image);

	//MANUAL THRESHOLDING
	int *sigma_thr;
	sigma_thr = manual_thresholding_sigma(sigma, imageW, imageH, SIGMA_THRESHOLD);
	
	//NORMALIZE IMAGE VALUES FROM 0-255 to 0-1 SINGLE PRECISION
	IplImage *canon_R, *canon_G, *canon_B;
	canon_R = cvCreateImage(cvSize(imageW, imageH), IPL_DEPTH_32F, 1);
	canon_G = cvCreateImage(cvSize(imageW, imageH), IPL_DEPTH_32F, 1);
	canon_B = cvCreateImage(cvSize(imageW, imageH), IPL_DEPTH_32F, 1);
	
	for (i = 0; i < imageH; i++) {
		for (j = 0; j < imageW; j++) {
			CV_IMAGE_ELEM(canon_R, float, i, j) = (float)((unsigned char)original_R->imageData[i*original_R->widthStep + j]) / 255;
			CV_IMAGE_ELEM(canon_G, float, i, j) = (float)((unsigned char)original_G->imageData[i*original_R->widthStep + j]) / 255;
			CV_IMAGE_ELEM(canon_B, float, i, j) = (float)((unsigned char)original_B->imageData[i*original_R->widthStep + j]) / 255;
		}
	}
		 
	//COMPUTING LAWS
	CvPoint offset;
	offset.x=2;
	offset.y=2;
	float *img_laws_R[14],*img_laws_G[14],*img_laws_B[14];
	IplImage *padded_R;
	IplImage *padded_G;
	IplImage *padded_B;

	padded_R=cvCreateImage(cvSize(original_Image->width+4,original_Image->height+4),IPL_DEPTH_32F,1);
	padded_G=cvCreateImage(cvSize(original_Image->width+4,original_Image->height+4),IPL_DEPTH_32F,1);
	padded_B=cvCreateImage(cvSize(original_Image->width+4,original_Image->height+4),IPL_DEPTH_32F,1);

	cvCopyMakeBorder(canon_R,padded_R,offset,IPL_BORDER_REPLICATE);
	cvCopyMakeBorder(canon_G,padded_G,offset,IPL_BORDER_REPLICATE);
	cvCopyMakeBorder(canon_B,padded_B,offset,IPL_BORDER_REPLICATE);
	
	for (k=0;k<14;k++) {
		img_laws_R[k]=(float *)malloc(imageW*imageH*sizeof(float));
		img_laws_G[k]=(float *)malloc(imageW*imageH*sizeof(float));
		img_laws_B[k]=(float *)malloc(imageW*imageH*sizeof(float));
		for (i=0;i<imageH;i++) {
			for (j=0;j<imageW;j++) {
				img_laws_R[k][i*imageW+j]=convolution_laws(padded_R,i,j,k);
				img_laws_G[k][i*imageW+j]=convolution_laws(padded_G,i,j,k);
				img_laws_B[k][i*imageW+j]=convolution_laws(padded_B,i,j,k);
			}
		}
	}


	for (k = 0; k < 14;k++) {
		for (i = 0; i < imageH*imageW; i++){
			if (img_laws_B[k][i] == -431602080.0) {
				j= 0;
			}
			if (img_laws_G[k][i] == -431602080.0) {
				j = 0;
			}
			if (img_laws_R[k][i] == -431602080.0) {
				j = 0;
			}
		}
	}


	//MEAN SHIFT
	float * im_arr;
	unsigned char *rgb_arr;
	im_arr=(float*)malloc(imageW*imageH*sizeof(float));
	rgb_arr=(unsigned char *)malloc(3*imageW*imageH*sizeof(unsigned char));
	float * laws_arr = (float *)malloc(14 * 3 * imageW*imageH*sizeof(float));
	for (i=0;i<imageH;i++) {
		for (j=0;j<imageW;j++) {
			im_arr[i*imageW+j]=(float)((unsigned char)original_Image_bw->imageData[i*original_Image_bw->widthStep+j]);\
			rgb_arr[i*imageW+3*j]=(unsigned char)original_R->imageData[i*original_R->widthStep+j];
			rgb_arr[i*imageW+3*j+1]=(unsigned char)original_G->imageData[i*original_R->widthStep+j];
			rgb_arr[i*imageW+3*j+2]=(unsigned char)original_B->imageData[i*original_R->widthStep+j];
			
			/*for (k = 0; k < 14; k++) {
				laws_arr[i*imageW * 42 + j * 42 + k + 0] = img_laws_R[k][i*imageW + j];
				laws_arr[i*imageW * 42 + j * 42 + k + 14] = img_laws_G[k][i*imageW + j];
				laws_arr[i*imageW * 42 + j * 42 + k + 28] = img_laws_B[k][i*imageW + j];
			}*/
		}
	}
	
	

	for (k = 0; k < 14; k++) {
		for (j = 0; j < imageW; j++) {
			for (i = 0; i < imageH; i++) {
				laws_arr[i*imageW*42 + j*42 + 3*k] = img_laws_R[k][i*imageW + j];
				laws_arr[i*imageW*42 + j*42 + 3*k+1] = img_laws_G[k][i*imageW + j];
				laws_arr[i*imageW*42 + j*42 + 3*k+2] = img_laws_B[k][i*imageW + j];
			}
		}
	}


	for (i = 0; i < imageH*imageW * 42; i++){
		if (laws_arr[i] == -431602080.0) {
			k = 0;
		}
	}

	int steps=2;
	unsigned int minArea=MINIMUM_SEGMANTATION_AREA;
	bool syn=true;
	unsigned int spBW=SPATIAL_BANDWITH;
	float fsBW=RANGE_BANDWITH;
	unsigned int w=(unsigned int)imageW;
	unsigned int h=(unsigned int)imageH;
	unsigned int grWin=2;
	unsigned int ii;
	float aij=(float)0.3;//mixture parameter
	float edgeThr=(float)0.3;
	int return_val;
    
	msImageProcessor ms;

	int N = 42;
	ms.DefineLInput(laws_arr, imageH, imageW, N);
	//int	N = 1;
	//ms.DefineLInput(im_arr, w, h, N);

    kernelType kern[2] = {DefaultKernelType, DefaultKernelType};
    int P[2] = {DefaultSpatialDimensionality, N};
    float tempH[2] = {1.0, 1.0};
	ms.DefineKernel(kern, tempH, P, 2);


    float * conf = NULL;//?
    float * grad = NULL;//?
    float * wght = NULL;//?
    
    if (syn) {
        /* perform synergistic segmentation */
        int maps_dim[2] = {w*h, 1};
        /* allcate memory for confidence and gradient maps */
        conf = (float *)malloc(w*h*sizeof(float));
		grad = (float *)malloc(w*h*sizeof(float));

        BgImage rgbIm;
        rgbIm.SetImage(rgb_arr, w, h, true);//false bw // true rgb!
        BgEdgeDetect edgeDetector(grWin);
        edgeDetector.ComputeEdgeInfo(&rgbIm, conf, grad);
        

        wght = (float *)malloc(w*h*sizeof(float));
        
        for ( ii = 0 ; ii < w*h; ii++ ) {
            wght[ii] = (grad[ii] > .002) ? aij*grad[ii]+(1-aij)*conf[ii] : 0;
        }
      ms.SetWeightMap(wght, edgeThr);
        if (ms.ErrorStatus)
            printf("edison_wraper:edison","Mean shift set weights: %s", ms.ErrorMessage);

    }
	ms.Filter(spBW, fsBW, MED_SPEEDUP);
    if (ms.ErrorStatus)
        printf("edison_wraper:edison","Mean shift filter: %s", ms.ErrorMessage);

    if (steps == 2) {
        ms.FuseRegions(fsBW, minArea);
        if (ms.ErrorStatus)
            printf("edison_wraper:edison","Mean shift fuse: %s", ms.ErrorMessage);
    }
	
	float * segmluv=(float *) malloc (N*imageH*imageW*sizeof(float));
	ms.GetRawData(segmluv); //LUV->RBG 1:1 if 1channel
	int * labels_out=NULL,*MPC_out=NULL;
	float *modes_out=NULL;
	int num_regions;
	num_regions=ms.GetRegions(&labels_out,&modes_out,&MPC_out);


	//---------------------NOT DOUBLE CHEKED CODE! USE WITH YOUR OWN RISK---------------------------------------//
	//CALCULATE THE CLASSIFICATION PROPERTIES avg_region_RGB avg_region_xy avg_region_laws density_edge
	//calculates the overall density of edges in the image
	unsigned char * canny=(unsigned char *)edge_Image->imageData;
	int tmp=0;
	float avg_density_edge;
	for (i=0;i<imageH;i++) {
		for (j=0;j<imageW;j++) {
			if (canny[i*edge_Image->widthStep+j]==255) {
				tmp++;
			}
		}
	}
	avg_density_edge=(float)tmp/(imageW*imageH);

	int evaluated_regions=0;
	int reg;
	float * region_density_edges=(float*)malloc(num_regions*sizeof(float));
	float * avg_region_R=(float*)malloc(num_regions*sizeof(float));
	float * avg_region_G=(float*)malloc(num_regions*sizeof(float));
	float * avg_region_B=(float*)malloc(num_regions*sizeof(float));
	float * avg_region_x=(float*)malloc(num_regions*sizeof(float));
	float * avg_region_y=(float*)malloc(num_regions*sizeof(float));

	int * edge_count=(int *)calloc (num_regions,sizeof(int));
	int * sum_R=(int *)calloc (num_regions,sizeof(int));
	int * sum_G=(int *)calloc (num_regions,sizeof(int));
	int * sum_B=(int *)calloc (num_regions,sizeof(int));
	int * sum_x=(int *)calloc (num_regions,sizeof(int));
	int * sum_y=(int *)calloc (num_regions,sizeof(int));
	int * regions_sharp_edges=(int *)calloc (num_regions,sizeof(int));
	int * regions_blurred_edges=(int *)calloc (num_regions,sizeof(int));
	float *sum_laws_R[14],*sum_laws_G[14],*sum_laws_B[14];
	float *avg_region_laws_R[14],*avg_region_laws_G[14],*avg_region_laws_B[14];

	for (i=0;i<14;i++) {
		sum_laws_R[i]=(float*)calloc(num_regions,sizeof(float));
		sum_laws_G[i]=(float*)calloc(num_regions,sizeof(float));
		sum_laws_B[i]=(float*)calloc(num_regions,sizeof(float));
		avg_region_laws_R[i]=(float*)malloc(num_regions*sizeof(float));
		avg_region_laws_G[i]=(float*)malloc(num_regions*sizeof(float));
		avg_region_laws_B[i]=(float*)malloc(num_regions*sizeof(float));
	}


	for (i=0;i<imageH;i++) {
		for (j=0;j<imageW;j++) {
			reg=labels_out[i*imageW+j];
			if (canny[i*edge_Image->widthStep+j]==255) {
				edge_count[reg]++;
				if (sigma[i*imageW+j]>SHARP_THRESHOLD) {
					regions_sharp_edges[reg]++;
				}
				else {
					regions_blurred_edges[reg]++;
				}
			}
			
			sum_R[reg]+=(uchar)original_R->imageData[i*original_R->widthStep+j];
			sum_G[reg]+=(uchar)original_G->imageData[i*original_R->widthStep+j];
			sum_B[reg]+=(uchar)original_B->imageData[i*original_R->widthStep+j];
			sum_x[reg]+=j;
			sum_x[reg]+=i;
			for (ii=0;ii<14;ii++) {
				sum_laws_R[ii][reg]+=img_laws_R[ii][i*imageW+j];
				sum_laws_G[ii][reg]+=img_laws_G[ii][i*imageW+j];
				sum_laws_B[ii][reg]+=img_laws_B[ii][i*imageW+j];
			}
		}
	}

	for (i=0;i<num_regions;i++) {
		region_density_edges[i]=(float)edge_count[i]/MPC_out[i];	
		avg_region_R[i]=(float)sum_R[i]/MPC_out[i];
		avg_region_G[i]=(float)sum_G[i]/MPC_out[i];
		avg_region_B[i]=(float)sum_B[i]/MPC_out[i];
		avg_region_x[i]=(float)sum_x[i]/MPC_out[i];
		avg_region_y[i]=(float)sum_x[i]/MPC_out[i];
		for (ii=0;ii<14;ii++) {
			avg_region_laws_R[ii][i]=(float)sum_laws_R[ii][i]/MPC_out[i];
			avg_region_laws_G[ii][i]=(float)sum_laws_G[ii][i]/MPC_out[i];
			avg_region_laws_B[ii][i]=(float)sum_laws_B[ii][i]/MPC_out[i];
		}
	}
	float threshold;
	if (0.9*avg_density_edge>0.015) {
			threshold=(float)0.9*avg_density_edge;
	}
	else {
		threshold=(float)0.015;
	}

	int *regions=(int *)malloc(num_regions*sizeof(int)); 
	for (i=0;i<num_regions;i++) {
		regions[i]=-1; //-1 uninitialized, 0 blurred, 1 sharp//
	}
	

	int point=0;
	int dim=48;
	int kneigh=3;
	ANNpointArray dataPts;
	dataPts=annAllocPts(1000,dim);
	int * points2regions=(int*)calloc(1000,sizeof(int));

	for (i=0;i<num_regions;i++) {
		if (MPC_out[i]>64 && region_density_edges[i]>threshold) {
			if ((float)regions_sharp_edges[i]/regions_blurred_edges[i]>10.0) {
				regions[i]=1;
				points2regions[point]=i;
				dataPts[point][0]=avg_region_x[i];
				dataPts[point][1]=avg_region_y[i];
				dataPts[point][2]=MPC_out[i];//region area
				dataPts[point][3]=avg_region_R[i];
				dataPts[point][4]=avg_region_G[i];
				dataPts[point][5]=avg_region_B[i];
				for (k=0;k<14;k++) {
					dataPts[point][6+k]=avg_region_laws_R[k][i];
				}
				for (k=0;k<14;k++) {
					dataPts[point][20+k]=avg_region_laws_G[k][i];
				}
				for (k=0;k<14;k++) {
					dataPts[point][34+k]=avg_region_laws_B[k][i];
				}
				point++;
			}
			else if ((float)regions_blurred_edges[i]/regions_sharp_edges[i]>10.0) {
				regions[i]=0;
				dataPts[point][0]=avg_region_x[i];
				dataPts[point][1]=avg_region_y[i];
				dataPts[point][2]=MPC_out[i];//region area
				dataPts[point][3]=avg_region_R[i];
				dataPts[point][4]=avg_region_G[i];
				dataPts[point][5]=avg_region_B[i];
				for (k=0;k<14;k++) {
					dataPts[point][6+k]=avg_region_laws_G[k][i];
				}
				for (k=0;k<14;k++) {
					dataPts[point][20+k]=avg_region_laws_G[k][i];
				}
				for (k=0;k<14;k++) {
					dataPts[point][34+k]=avg_region_laws_B[k][i];
				}
				point++;
			}
		}
	}
	ANNkd_tree*	kdTree;	
	kdTree= new ANNkd_tree(dataPts,point,dim);
	ANNidxArray	nnIdx;
	nnIdx = new ANNidx[kneigh];
	ANNdistArray dists;
	dists = new ANNdist[kneigh];	
	ANNpoint queryPt;
	queryPt = annAllocPt(dim);	

	for (i=0;i<num_regions;i++) {
		if (regions[i]==-1) {
			queryPt[0]=avg_region_x[i];
			queryPt[1]=avg_region_y[i];
			queryPt[2]=MPC_out[i];//region area
			queryPt[3]=avg_region_R[i];
			queryPt[4]=avg_region_G[i];
			queryPt[5]=avg_region_B[i];
			for (k=0;k<14;k++) {
				queryPt[6+k]=avg_region_laws_G[k][i];
			}
			for (k=0;k<14;k++) {
				queryPt[20+k]=avg_region_laws_G[k][i];
			}
			for (k=0;k<14;k++) {
				queryPt[34+k]=avg_region_laws_B[k][i];
			}
			kdTree->annkSearch(queryPt,kneigh,nnIdx,dists);
			tmp=0;
			for (k=0;k<kneigh;k++) {
				tmp+=regions[points2regions[nnIdx[k]]];
			}
			if (tmp>kneigh/2) {
				regions[i]=1;
			}
			else {
				regions[i]=0;
			}
		}
	}
	cvSaveImage("ResultCanny.jpg",edge_Image);
	int region;
	for (i=0;i<imageH;i++) {
		for (j=0;j<imageW;j++) {
//			derivative_Rx->imageData[i*derivative_Rx->widthStep+j]=(char)segmluv[i*imageW+j];
			region=labels_out[i*imageW+j];
			if (regions[region]==1) {
				edge_Image->imageData[i*edge_Image->width+j]=-1;
			}
			else {
				edge_Image->imageData[i*edge_Image->width+j]=0;
			}
		}
	}

	cvSaveImage("final_selection.jpg",edge_Image);



//release kai save
return 0;
}
