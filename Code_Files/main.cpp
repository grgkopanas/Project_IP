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
#define SPATIAL_BANDWITH 6
#define RANGE_BANDWITH 6
#define MINIMUM_SEGMANTATION_AREA 50
#define FILENAME "DSC_0027.jpg"

struct g {
	double gxx;
	double gyy;
	double gxy;
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

int kernels[14][5][5]={
					   {{-2,-6,-6,-2,0},{-6,-16,-12,0,3},{-6,-12,0,12,6},{-2,0,12,16,6},{0,2,6,6,2}},//E5L5 + L5E5
					   {{-2,-4,-4,-4,-2},{-4,0,8,0,-4},{-4,8,24,8,-4},{-4,0,0,0,-4},{-2,-4,-4,-4,-2}},//S5L5 + L5S5
					   {{-2,-2,-6,-6,0},{-2,16,12,0,6},{-6,12,0,-12,6},{-6,0,-12,-16,2},{0,6,6,2,2}},//W5L5 + L5W5
					   {{2,0,12,0,2},{0,-32,0,-32,0},{12,0,72,0,12},{0,-32,0,-32,0},{2,0,12,0,2}},//R5L5 + L5R5
					   {{1,2,-2,-2,-2},{2,0,-4,0,-2},{-2,-4,0,4,2},{-2,0,4,0,2},{0,2,2,-2,0}},//S5E5 + E5S5
					   {{2,0,0,0,-2},{0,-8,0,8,0},{0,0,0,0,0},{0,8,0,-8,0},{-2,0,0,0,2}},//W5E5 +E5W5
					   {{-2,2,-6,6,0},{2,16,-12,0,-6},{-6,-12,0,12,6},{6,0,12,-16,-2},{0,-6,6,-2,2}},//R5E5 + E5R5
					   {{2,-2,-2,2,0},{-2,0,4,0,-2},{-2,4,0,-4,2},{2,0,-4,0,2},{0,-2,2,2,-2}},//W5S5 + S5W5
					   {{-2,4,-4,4,0},{4,0,-8,0,4},{-4,0,4,-8,-4},{4,0,-8,0,4},{-2,4,-4,4,0}},//R5S5 + S5R5
					   {{-2,6,-6,2,0},{6,-16,12,0,-2},{-6,12,0,-12,6},{2,0,-12,16,-6},{0,-2,6,-6,2}},//R5W5 + W5R5
					   {{2,4,0,-4,-2},{4,8,0,-8,-4},{0,0,0,0,0},{-4,-8,0,8,4},{-2,-4,0,4,2}},//2*E5E5
					   {{2,0,-4,0,1},{0,0,0,0,0},{-4,0,8,0,-4},{0,0,0,0,0},{2,0,-4,0,2}},//2*S5S5
					   {{2,-4,0,4,-2},{-4,8,0,-8,4},{0,0,0,0,0},{4,-8,0,8,-4},{-2,4,0,-4,2}},//2*W5W5
					   {{2,-8,12,-8,2},{-8,32,-48,32,-8},{12,-48,64,-48,12},{-8,32,-48,32,-8},{2,-8,12,-8,2}}//2*R5R5
					   };

//To DiZenzo de doulevei kala gia kapoion logo
//#define DIZENZO_DERIVATIVE
#define SOBEL_DERIVATIVE
#define DISTANCE_FROM_CANNY 5

sigma * edge_sharpness(	IplImage *derivative_Rx,IplImage *derivative_Gx,
						IplImage *derivative_Bx,IplImage *derivative_Ry,
						IplImage *derivative_Gy,IplImage *derivative_By,
						IplImage *edge_Image);
int checkneighbors(unsigned char * canny, unsigned char * weakEdges, unsigned char *visited, int i, int j, int imageW, int imageH);
int myCanny(IplImage *src, unsigned char * canny_image);
int convolution2d(IplImage *padded_32b_R,int i,int j,int laws_num);
int connectgaps(unsigned char * img, int imageW, int imageH);

sigma * edge_sharpness(IplImage *derivative_Rx,IplImage *derivative_Gx,IplImage *derivative_Bx,
					   IplImage *derivative_Ry,IplImage *derivative_Gy,
					   IplImage *derivative_By,IplImage *edge_Image) {
	
	unsigned char rx,gx,bx,ry,gy,by;
	float gxx,gyy,gxy,gtheta_a,gtheta_b,theta;
	struct g *image_g;
	int i,j,k;

	int imageW=edge_Image->width;
	int imageH=edge_Image->height;
	int widthStep=edge_Image->widthStep;

	image_g=(struct g *)malloc(imageW*imageH*sizeof(struct g));


	for (i=0;i<imageH;i++) {
		for (j=0;j<imageW;j++) {

			rx=(unsigned char)derivative_Rx->imageData[i*widthStep+j];
			gx=(unsigned char)derivative_Gx->imageData[i*widthStep+j];
			bx=(unsigned char)derivative_Bx->imageData[i*widthStep+j];
			ry=(unsigned char)derivative_Ry->imageData[i*widthStep+j];
			gy=(unsigned char)derivative_Gy->imageData[i*widthStep+j];
			by=(unsigned char)derivative_By->imageData[i*widthStep+j];
			gxx=(float)(rx*rx + gx*gx + bx*bx);
			gyy=(float)(ry*ry + gy*gy + by*by);
			gxy=(float)(rx*ry + gx*gy + rx*ry);

			if ( gxx == 0.0 && gyy == 0.0 ) {
				theta=0.0;
			}
			else {
				theta=(float)(0.5 * atan( (2*gxy) / (gxx - gyy)));
			}
			gtheta_a =(float)	0.5 * ((gxx + gyy) + (gxx - gyy)*
								cos(2 * theta) + 2 * gxy * sin(2 * theta)); 
			gtheta_b =(float)	(0.5 * ((gxx + gyy) + (gxx - gyy)*
								cos(2 * (theta+PI/2)) + 2*gxy*sin(2 * (theta+PI/2))));
			image_g[i*imageW+j].gxx=sqrt(gxx);
			image_g[i*imageW+j].gyy=sqrt(gyy);
			image_g[i*imageW+j].gxy=sqrt(gxy);
			if (gtheta_a>gtheta_b) {
				image_g[i*imageW+j].gtheta=sqrt(gtheta_a);
			}
			else {
				image_g[i*imageW+j].gtheta=sqrt(gtheta_b);
			}
			image_g[i*imageW+j].theta=theta;
		}
	}

	float max_value;
	int max_idx;
	int left_end,right_end,up_end,down_end;
	int tmp1,tmp2;
	lamda *l=(lamda *)malloc(imageW*imageH*sizeof(lamda));

	for (i=0;i<imageH;i++) {
		for (j=0;j<imageW;j++) {
			if ((unsigned char)edge_Image->imageData[i*widthStep+j]==255) {
				if (image_g[i*imageW+j].gxx>image_g[i*imageW+j].gyy) { //ipologizoume to l kata x else kata y			
					//vriskoume to megisto tou edge se euros 10 5-aristera-5 deksia v2.0
					max_value=image_g[i*imageW+j].gxx;
					max_idx=j;
					if (j-DISTANCE_FROM_CANNY<0) {
						tmp1=0;
					}
					else {
						tmp1=j-DISTANCE_FROM_CANNY;
					}
					if (j+DISTANCE_FROM_CANNY+1>imageW-1) {
						tmp2=imageW-1;
					}
					else {
						tmp2=j+DISTANCE_FROM_CANNY+1;
					}

					for (k=tmp1;k<tmp2;k++) {
						if (image_g[i*imageW+k].gxx>max_value) {
							max_value=image_g[i*imageW+k].gxx;
							max_idx=k;
						}
					}
					if (max_idx==tmp2) {
						if (tmp2!=imageW) {
							if (image_g[i*imageW+max_idx].gxx<image_g[i*imageW+max_idx+1].gxx) {
								edge_Image->imageData[i*edge_Image->widthStep+j]=0; //itan canny error opote to ekana 0
							}
						}
					}
					else if (max_idx==tmp1) {
						if (tmp1!=0) {
							if (image_g[i*imageW+max_idx].gxx<image_g[i*imageW+max_idx-1].gxx) {
								edge_Image->imageData[i*edge_Image->widthStep+j]=0; //itan canny error opote to ekana 0
							}
						}
					}


					left_end=max_idx-1;
					if (left_end<0) left_end=0;
					right_end=max_idx+1;
					if (right_end>imageW-1) right_end=imageW-1;
					if (left_end>0) {
						while (!(image_g[i*imageW+left_end].gxx<0.1*max_value) &&  
							   !(image_g[i*imageW+left_end].gxx-image_g[i*imageW+left_end-1].gxx>0.05*image_g[i*imageW+left_end].gxx)&&left_end>1) {
		
							left_end--;
						}
					}
					if (right_end<imageW-1) {
						while (!(image_g[i*imageW+right_end].gxx<0.1*max_value) &&  
						!(image_g[i*imageW+right_end].gxx-image_g[i*imageW+right_end+1].gxx>0.05*image_g[i*imageW+right_end].gxx)&&right_end<imageW-2) {
	
							right_end++;
						}
					}
					l[i*imageW+j].l=right_end-left_end;
					l[i*imageW+j].small_end=left_end;
					l[i*imageW+j].big_end=right_end;
					l[i*imageW+j].dir=0;
				}
				else { //gyy>gxx opote psaxnw katheta
					max_value=image_g[i*imageW+j].gyy;
					max_idx=i;
					
					if (i-DISTANCE_FROM_CANNY<0) {
						tmp1=0;
					}
					else {
						tmp1=i-DISTANCE_FROM_CANNY;
					}

					if (i+DISTANCE_FROM_CANNY+1>imageH-1) {
						tmp2=imageH-1;
					}
					else {
						tmp2=i+DISTANCE_FROM_CANNY+1;
					}

					for (k=tmp1;k<tmp2;k++) {
						if (image_g[k*imageW+j].gyy>max_value) {
							max_value=image_g[k*imageW+j].gyy;
							max_idx=k;
						}
					}

					if (max_idx==tmp2) {
						if (tmp2!=imageH) {
							if (image_g[max_idx*imageW+j].gyy<image_g[(max_idx+1)*imageW+j].gyy) {
								edge_Image->imageData[i*edge_Image->widthStep+j]=0; //itan canny error opote to ekana 0
							}
						}
					}
					else if (max_idx==tmp1) {
						if (tmp1!=0) {
							if (image_g[max_idx*imageW+j].gyy<image_g[(max_idx-1)*imageW+j].gyy) {
								edge_Image->imageData[i*edge_Image->widthStep+j]=0; //itan canny error opote to ekana 0
							}
						}
					}

					up_end=max_idx-1;
					if (up_end<0) up_end=0;
					down_end=max_idx+1;
					if (down_end>imageH-1) down_end=imageH-1; 
					if ( (up_end>0) ) { //prostateyw ton prwto elegxo sto (up_end-1)*imageW+j
						while (!(image_g[up_end*imageW+j].gyy<0.1*max_value) &&  
							   !(image_g[up_end*imageW+j].gyy-image_g[(up_end-1)*imageW+j].gyy>0.05*image_g[up_end*imageW+j].gyy)&&up_end>1) { //>1 gia prosatsia sto (up_end-1)*imageW+j
		
							up_end--;
						}
					}
					if ( (down_end<imageH-1)) { //prostateyw ton prwto elegxo sto (down_end+1)*imageW+j
						while (!(image_g[down_end*imageW+i].gyy<0.1*max_value) &&  
							!(image_g[down_end*imageW+j].gyy-image_g[(down_end+1)*imageW+j].gyy>0.05*image_g[down_end*imageW+j].gyy)&&down_end<imageH-2) {//-2 gia prostasia sto (down_end+1)*imageW+j
	
							down_end++;
						}
					}
					l[i*imageW+j].l=down_end-up_end;
					l[i*imageW+j].small_end=up_end;
					l[i*imageW+j].big_end=down_end;
					l[i*imageW+j].dir=1;
				}
			}
			else { //canny=0
				l[i*imageW+j].l=0;
				l[i*imageW+j].dir=0;
				l[i*imageW+j].small_end=0;
				l[i*imageW+j].big_end=0;
			}
		}
		
		

	}

	float *l_theta=(float *) malloc (imageH*imageW*(sizeof(float)));

	for (i=0;i<imageH;i++) {
		for (j=0;j<imageW;j++) {
			
			if (l[i*imageW+j].dir==0) {//xx>yy
				l_theta[i*imageW+j]=(image_g[i*imageW+j].gxx/image_g[i*imageW+j].gtheta)*l[i*imageW+j].l;
			}
			else { //yy>xx
				l_theta[i*imageW+j]=(image_g[i*imageW+j].gyy/image_g[i*imageW+j].gtheta)*l[i*imageW+j].l;
			}

			if (image_g[i*imageW+j].gtheta==0) {
				l_theta[i*imageW+j]=0;
			}
		}
	}

	
	float s;
	sigma *s_theta=(sigma *)malloc(imageW*imageH*sizeof(sigma));
	float sum1,sum2;

	for (i=0;i<imageH;i++) {
		for (j=0;j<imageW;j++) {
			s_theta[i*imageW+j].s=0.0;
			s_theta[i*imageW+j].dir=0;
		}
	}

	for (i=1;i<imageH-1;i++) {//den exei noima na parw ta boundaries afou ta gxx kai gyy einai 0 even if its is a canny point
		for (j=1;j<imageW-1;j++) {
			sum1=0;
			sum2=0;
			if ((unsigned char)edge_Image->imageData[i*widthStep+j]==0) {
				s_theta[i*imageW+j].s=0.0;
				s_theta[i*imageW+j].dir=0;
			}
			else if (image_g[i*imageW+j].gxx>=image_g[i*imageW+j].gyy) {
				for (k=l[i*imageW+j].small_end;k<l[i*imageW+j].big_end;k++){
					sum1=sum1+image_g[i*imageW+k].gxx*l[i*imageW+k].l*l[i*imageW+k].l;
					sum2=sum2+image_g[i*imageW+j].gxx;
					
				}
				s=sqrt(sum1/sum2);
				s_theta[i*imageW+j].s=(image_g[i*imageW+j].gxx/image_g[i*imageW+j].gtheta)*s;
				s_theta[i*imageW+j].dir=0;
			}
			else if (image_g[i*imageW+j].gxx<image_g[i*imageW+j].gyy) {
				for (k=l[i*imageW+j].small_end;k<l[i*imageW+j].big_end;k++){
					sum1=sum1+image_g[k*imageW+j].gyy*l[k*imageW+j].l*l[k*imageW+j].l;
					sum2=sum2+image_g[k*imageW+j].gyy;
					
				}
				s=sqrt(sum1/sum2);
				s_theta[i*imageW+j].s=(image_g[i*imageW+j].gyy/image_g[i*imageW+j].gtheta)*s;
				s_theta[i*imageW+j].dir=1;
			}
		}
	}
	//for debugging purposes only
		for (i=0;i<imageH;i++) {
			for (j=0;j<imageW;j++) {
				if (edge_Image->imageData[edge_Image->widthStep*i+j]==-1) {
					if (image_g[i*imageW+j].gxx+image_g[i*imageW+j].gxy+image_g[i*imageW+j].gyy==0) {
						printf("Potition %d %d\n",i,j);
					}
				}
			}
		}
	return s_theta;
}

int convolution2d(IplImage *padded_32b,int posy,int posx,int laws_num) {

	int i,j;
	int res;
	res=0;
	int widthStep=padded_32b->widthStep;

	for (i = -2; i <= 2; i++) {
		for (j = -2; j <= 2; j++) {
			res += padded_32b->imageData[(posy + 2 + i)*widthStep + posx + 2 + j] * kernels[laws_num][i+2][j+2];
		}
	}

	return res;
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

	//smoothing before canny, matlab does it automatically
	//cvSmooth(original_Image_bw,gaussian_image_bw,CV_GAUSSIAN,7,7,3);
	//cvCanny(gaussian_image_bw,edge_Image,CANNY_THRESHOLD_1,CANNY_THRESHOLD_2,3);
	unsigned char * canny_im = (unsigned char *)calloc(imageW*imageH,sizeof(unsigned char));
	myCanny(original_Image, canny_im);
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
	cvSobel(original_R,derivative_Rx,1,0,3);
	cvSobel(original_B,derivative_Bx,1,0,3);
	cvSobel(original_G,derivative_Gx,1,0,3);

	cvSobel(original_R,derivative_Ry,0,1,3);
	cvSobel(original_B,derivative_By,0,1,3);
	cvSobel(original_G,derivative_Gy,0,1,3);

	cvSaveImage("Rx.jpg",derivative_Rx);
	cvSaveImage("Bx.jpg",derivative_Bx);
	cvSaveImage("Gx.jpg",derivative_Gx);
#endif 

#ifdef DIZENZO_DERIVATIVE
	for (i=0;i<imageH;i++) {
		for (j=0;j<imageW;j++) {
			derivative_Rx->imageData[i*widthStep+j]=0;
			derivative_Gx->imageData[i*widthStep+j]=0;
			derivative_Bx->imageData[i*widthStep+j]=0;

			derivative_Ry->imageData[i*widthStep+j]=0;
			derivative_Gy->imageData[i*widthStep+j]=0;
			derivative_By->imageData[i*widthStep+j]=0;
		}
	}

	for (i=1;i<imageH-1;i++) {
		for (j=1;j<imageW-1;j++) {
				derivative_Rx->imageData[i*widthStep+j] = ((uchar)original_R->imageData[i*widthStep + (j+1)] +  (uchar)original_R->imageData[(i+1)*widthStep + (j+1)]
														  - (uchar)original_R->imageData[i*widthStep + j] - (uchar)original_R->imageData[(i+1)*widthStep + j])/2;

				derivative_Bx->imageData[i*widthStep+j] = ((uchar)original_B->imageData[i*widthStep + (j+1)] +  (uchar)original_B->imageData[(i+1)*widthStep + (j+1)]
														  - (uchar)original_B->imageData[i*widthStep + j] - (uchar)original_B->imageData[(i+1)*widthStep + j])/2;

				derivative_Gx->imageData[i*widthStep+j] = ((uchar)original_G->imageData[i*widthStep + (j+1)] + (uchar) original_G->imageData[(i+1)*widthStep + (j+1)]
														  - (uchar)original_G->imageData[i*widthStep + j] - (uchar)original_G->imageData[(i+1)*widthStep + j])/2;


				derivative_Ry->imageData[i*widthStep+j] = ((uchar)original_R->imageData[(i+1)*widthStep + j] +  (uchar)original_R->imageData[(i+1)*widthStep + (j+1)]
														  - (uchar)original_R->imageData[i*widthStep + j] - (uchar)original_R->imageData[i*widthStep + j+1])/2;

				derivative_By->imageData[i*widthStep+j] = ((uchar)original_B->imageData[(i+1)*widthStep + j] +  (uchar)original_B->imageData[(i+1)*widthStep + (j+1)]
														  - (uchar)original_B->imageData[i*widthStep + j] - (uchar)original_B->imageData[i*widthStep + j+1])/2;

				derivative_Gy->imageData[i*widthStep+j] = ((uchar)original_G->imageData[(i+1)*widthStep + j] +  (uchar)original_G->imageData[(i+1)*widthStep + (j+1)]
														  - (uchar)original_G->imageData[i*widthStep + j] - (uchar)original_G->imageData[i*widthStep + j+1])/2;

		}
	}
#endif

	sigma *s_theta;
	s_theta=edge_sharpness(derivative_Rx,derivative_Gx,derivative_Bx,derivative_Ry,derivative_Gy,derivative_By,edge_Image);
	
	//COMPUTING LAWS

	CvPoint offset;
	offset.x=2;
	offset.y=2;
	int *img_laws_R[14],*img_laws_G[14],*img_laws_B[14];
	IplImage *padded_R;
	IplImage *padded_G;
	IplImage *padded_B;

	
	padded_R=cvCreateImage(cvSize(original_Image->width+4,original_Image->height+4),8,1);
	padded_G=cvCreateImage(cvSize(original_Image->width+4,original_Image->height+4),8,1);
	padded_B=cvCreateImage(cvSize(original_Image->width+4,original_Image->height+4),8,1);

	cvCopyMakeBorder(original_R,padded_R,offset,IPL_BORDER_CONSTANT,cvScalarAll(0));
	cvCopyMakeBorder(original_G,padded_G,offset,IPL_BORDER_CONSTANT,cvScalarAll(0));
	cvCopyMakeBorder(original_B,padded_B,offset,IPL_BORDER_CONSTANT,cvScalarAll(0));
	
	for (k=0;k<14;k++) {
		img_laws_R[k]=(int *)malloc(imageW*imageH*sizeof(int));
		img_laws_G[k]=(int *)malloc(imageW*imageH*sizeof(int));
		img_laws_B[k]=(int *)malloc(imageW*imageH*sizeof(int));
		for (i=0;i<imageH;i++) {
			for (j=0;j<imageW;j++) {
				img_laws_R[k][i*imageW+j]=convolution2d(padded_R,i,j,k);
				img_laws_G[k][i*imageW+j]=convolution2d(padded_G,i,j,k);
				img_laws_B[k][i*imageW+j]=convolution2d(padded_B,i,j,k);
			}
		}
	}

	//MEAN SHIFT
	float * im_arr;
	unsigned char *rgb_arr;
	im_arr=(float*)malloc(imageW*imageH*sizeof(float));
	rgb_arr=(unsigned char *)malloc(3*imageW*imageH*sizeof(unsigned char));
	for (i=0;i<imageH;i++) {
		for (j=0;j<imageW;j++) {
			im_arr[i*imageW+j]=(float)((unsigned char)original_Image_bw->imageData[i*original_Image_bw->widthStep+j]);\
				rgb_arr[i*imageW+3*j]=(unsigned char)original_R->imageData[i*original_R->widthStep+j];
				rgb_arr[i*imageW+3*j+1]=(unsigned char)original_G->imageData[i*original_R->widthStep+j];
				rgb_arr[i*imageW+3*j+2]=(unsigned char)original_B->imageData[i*original_R->widthStep+j];
		}
	}

	msImageProcessor ms;
	ms.DefineLInput(im_arr,imageH,imageW,1);

	int steps=2;
	unsigned int minArea=MINIMUM_SEGMANTATION_AREA;
	unsigned int N=1; 
	bool syn=true;
	unsigned int spBW=SPATIAL_BANDWITH;
	float fsBW=RANGE_BANDWITH;
	unsigned int w=(unsigned int)imageW;
	unsigned int h=(unsigned int)imageH;
	unsigned int grWin=2;
	unsigned int ii;
	float aij=(float)0.3;//mixture parameter
	float edgeThr=(float)0.3;
    
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
	
	float * segmluv=(float *) malloc (imageH*imageW*sizeof(float));
	ms.GetRawData(segmluv); //LUV->RBG 1:1 if 1channel
	int * labels_out=NULL,*MPC_out=NULL;
	float *modes_out=NULL;
	int num_regions;
	num_regions=ms.GetRegions(&labels_out,&modes_out,&MPC_out);

	//CALCULATE THE CLASSIFICATION PROPERTIES avg_region_RGB avg_region_xy avg_region_laws density_edge
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
	int *sum_laws_R[14],*sum_laws_G[14],*sum_laws_B[14];
	float *avg_region_laws_R[14],*avg_region_laws_G[14],*avg_region_laws_B[14];

	for (i=0;i<14;i++) {
		sum_laws_R[i]=(int*)calloc(num_regions,sizeof(int));
		sum_laws_G[i]=(int*)calloc(num_regions,sizeof(int));
		sum_laws_B[i]=(int*)calloc(num_regions,sizeof(int));
		avg_region_laws_R[i]=(float*)malloc(num_regions*sizeof(float));
		avg_region_laws_G[i]=(float*)malloc(num_regions*sizeof(float));
		avg_region_laws_B[i]=(float*)malloc(num_regions*sizeof(float));
	}


	for (i=0;i<imageH;i++) {
		for (j=0;j<imageW;j++) {
			reg=labels_out[i*imageW+j];
			if (canny[i*edge_Image->widthStep+j]==255) {
				edge_count[reg]++;
				if (s_theta[i*imageW+j].s>SHARP_THRESHOLD) {
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
			derivative_Rx->imageData[i*derivative_Rx->widthStep+j]=(char)segmluv[i*imageW+j];
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


double convolution2d(IplImage *buffer, int posy, int posx, double *mask, int type) {
	int i, j;
	double res;

	res = 0;

	if (type == 0) {
		for (i = -3; i <= 3; i++) {
			for (j = -3; j <= 3; j++) {
				res += CV_IMAGE_ELEM(buffer, unsigned char, posy + i, posx + j)* mask[(i + 3) * 7 + j + 3];
			}
		}
	}

	if (type == 1) {
		for (i = -3; i <= 3; i++) {
			for (j = -3; j <= 3; j++) {
				res += CV_IMAGE_ELEM(buffer, double, posy + i, posx + j)* mask[(i + 3) * 7 + j + 3];
			}
		}
	}


	return (res);
}


void filter_F_2d(IplImage *img, double *dst, double *kernel,int type) {
	int i, j;
	int imageW = img->width;
	int imageH = img->height;
	IplImage *img_border;

	if (type == 0) {
		img_border = cvCreateImage(cvSize(img->width + 6, img->height + 6), 8, 1);
	}
	else {
		img_border = cvCreateImage(cvSize(img->width + 6, img->height + 6), IPL_DEPTH_64F, 1);
	}

	CvPoint b;
	b.x = 3;
	b.y = 3;
	cvCopyMakeBorder(img, img_border, b, IPL_BORDER_REPLICATE);

	for (i = 0; i < imageH; i++){
		for (j = 0; j < imageW; j++) {
			dst[i*imageW + j] = convolution2d(img_border, i+3, j+3, kernel,type);
		}
	}
}

int myCanny(IplImage *src, unsigned char * canny_image) {

	IplImage *src_gau_R, *src_gau_G, *src_gau_B, *src_gau;
	int i, j;

	src_gau = cvCreateImage(cvSize(src->width, src->height), 8, 3);
	src_gau_R = cvCreateImage(cvSize(src->width, src->height), 8, 1);
	src_gau_G = cvCreateImage(cvSize(src->width, src->height), 8, 1);
	src_gau_B = cvCreateImage(cvSize(src->width, src->height), 8, 1);
	int imageW = src->width;
	int imageH = src->height;

	//cvSmooth(src, src_gau, CV_GAUSSIAN, 7, 7, 3);
	cvSplit(src, src_gau_B, src_gau_G, src_gau_R, NULL);

	
	//my gaussian filter 2-d
	double *gau = (double *)malloc(7 * 7 * sizeof(double));

	double sig = 0.7;
	double tmp;
	for (i = -3; i < 4; i++) {
		for (j = -3; j <= 3; j++) {
			tmp = exp((double)(-i*i - j*j) / (2 * sig*sig));
			gau[(i + 3) * 7 + j + 3] = tmp / ((2 * PI*sig*sig)*(2 * PI*sig*sig));
		}
	}
	double * R_blurred = (double *)malloc(imageW*imageH*sizeof(double));
	double * G_blurred = (double *)malloc(imageW*imageH*sizeof(double));
	double * B_blurred = (double *)malloc(imageW*imageH*sizeof(double));

	filter_F_2d(src_gau_R, R_blurred, gau,0);
	filter_F_2d(src_gau_G, G_blurred, gau,0);
	filter_F_2d(src_gau_B, B_blurred, gau,0);

	IplImage *R_blurred_double = cvCreateImage(cvSize(imageW,imageH), IPL_DEPTH_64F, 1);
	IplImage *G_blurred_double = cvCreateImage(cvSize(imageW, imageH), IPL_DEPTH_64F, 1);
	IplImage *B_blurred_double = cvCreateImage(cvSize(imageW, imageH), IPL_DEPTH_64F, 1);

	for (i = 0; i < imageH; i++){
		for (j = 0; j < imageW; j++) {
			CV_IMAGE_ELEM(R_blurred_double, double, i, j) = R_blurred[i*imageW + j];
			CV_IMAGE_ELEM(G_blurred_double, double, i, j) = G_blurred[i*imageW + j];
			CV_IMAGE_ELEM(B_blurred_double, double, i, j) = B_blurred[i*imageW + j];
		}
	}
	for (i = -3; i <= 3; i++) {
		for (j = -3; j <= 3; j++) {
			tmp = exp((double)(-i*i - j*j) / (2 * sig*sig));
			gau[(i + 3) * 7 + j + 3] = - i * tmp / (PI*sig*sig);
		}
	}

	double * Ry = (double *)malloc(imageW*imageH*sizeof(double));
	double * Gy = (double *)malloc(imageW*imageH*sizeof(double));
	double * By = (double *)malloc(imageW*imageH*sizeof(double));
	double * Rx = (double *)malloc(imageW*imageH*sizeof(double));
	double * Gx = (double *)malloc(imageW*imageH*sizeof(double));
	double * Bx = (double *)malloc(imageW*imageH*sizeof(double));

	filter_F_2d(R_blurred_double, Ry, gau, 1);
	filter_F_2d(G_blurred_double, Gy, gau, 1);
	filter_F_2d(B_blurred_double, By, gau, 1);

	for (i = -3; i <= 3; i++) {
		for (j = -3; j <= 3; j++) {
			tmp = exp((double)(-j*j - i*i) / (2 * sig*sig));
			gau[(i + 3) * 7 + j + 3] = -j * tmp / (PI*sig*sig);
		}
	}

	filter_F_2d(R_blurred_double, Rx, gau, 1);
	filter_F_2d(G_blurred_double, Gx, gau, 1);
	filter_F_2d(B_blurred_double, Bx, gau, 1);
	
	struct g * image_g = (struct g *)malloc(imageW*imageH*sizeof(struct g));
	double rx, gx, bx, ry, gy, by;
	double gxx, gyy, gxy, theta, gtheta_a, gtheta_b;
	double temp;


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
			gxy = (rx*ry + gx*gy + rx*ry);

			if (gxx != gxy){
				temp = (double)(2 * gxy) / (gxx - gyy);
				theta = 0.5 * (atan(temp));
			}
			else {
				theta = PI / 4;
			}

			gtheta_a = 0.5 * ((gxx + gyy) + (gxx - gyy)*
				cos(2 * theta) + 2 * gxy * sin(2 * theta));

			gtheta_b = 0.5 * ((gxx + gyy) + (gxx - gyy)*
				cos(2 * (theta + PI / 2)) + 2 * gxy*sin(2 * (theta + PI / 2)));

			gtheta_a = sqrt(abs(gtheta_a));
			gtheta_b = sqrt(abs(gtheta_b));

			if (gtheta_a > gtheta_b) {
				image_g[i*imageW + j].gtheta = gtheta_a;
				image_g[i*imageW + j].magn_x = gtheta_a * cos(theta);
				image_g[i*imageW + j].magn_y = gtheta_a * sin(theta);
				image_g[i*imageW + j].theta = theta;
			}
			else {
				image_g[i*imageW + j].gtheta = gtheta_b;
				image_g[i*imageW + j].magn_x = (gtheta_b * cos(theta + PI / 2));
				image_g[i*imageW + j].magn_y = (gtheta_b * sin(theta + PI / 2));
				image_g[i*imageW + j].theta =  (theta + PI / 2);
			}
		}
	}

	//Normalize gtheta 0-1
	double gmax = 0;
	double gmin = 3.0;
	int column, row;
	for (i = 0; i < imageH; i++) {
		for (j = 0; j < imageW; j++) {
			if (image_g[i*imageW+j].gtheta>gmax) {
				column = j;
				row = i;
				gmax = image_g[i*imageW + j].gtheta;
			}
			if (image_g[i*imageW + j].gtheta<gmin) {
				gmin = image_g[i*imageW + j].gtheta;
			}
		}
	}

	for (i = 0; i < imageH; i++){
		for (j = 0; j < imageW; j++){
			image_g[i*imageW + j].gtheta = (image_g[i*imageW + j].gtheta - gmin) / (gmax - gmin);
		}
	}

	float ay, ax, d, grad1, grad2;
	int flag;
	unsigned char * localMax = (unsigned char *)calloc(imageW*imageH, sizeof(unsigned char));
	unsigned char * weakEdges = (unsigned char *)calloc(imageW*imageH, sizeof(unsigned char));
	unsigned char * strongEdges = (unsigned char *)calloc(imageW*imageH, sizeof(unsigned char));
	for (i = 1; i < imageH - 1; i++) {
		for (j = 1; j < imageW - 1; j++)	{
			ax = image_g[i*imageW + j].magn_x;
			ay = image_g[i*imageW + j].magn_y;
			flag = 0;
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

	for (i = 0; i < imageH; i++) {
		for (j = 0; j < imageW; j++) {
			if (strongEdges[i*imageW + j] == 1) {
				src_gau_R->imageData[i*src_gau_R->widthStep + j] = -1;
			}
			else {
				src_gau_R->imageData[i*src_gau_R->widthStep + j] = 0;
			}
			if (weakEdges[i*imageW + j] == 1) {
				src_gau_G->imageData[i*src_gau_G->widthStep + j] = -1;
			}
			else {
				src_gau_G->imageData[i*src_gau_G->widthStep + j] = 0;
			}
		}
	}
	cvSaveImage("strongEdges.jpg", src_gau_R);
	cvSaveImage("weakEdges.jpg", src_gau_G);
	unsigned char * visited = (unsigned char *)calloc(imageW*imageH, sizeof(unsigned char));

	//make weakedges 0 in borders
	for (i = 0; i < imageH; i++) {
		for (j = 0; j < imageW; j++) {
			if (i == 0 || j == 0 || i == imageH - 1 || j == imageW - 1){
				weakEdges[i*imageW + j] = 0;
				strongEdges[i*imageW + j] = 0;
			}
		}
	}
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
	}

	//edges are all the weak edges that are 8-connected with a strong edge
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
	for (i = 1; i < imageH-1; i++){
		for (j = 1; j < imageW-1; j++) {
			flag = 0;
			for (k= -1; k <= 1; k++) {
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

	cvReleaseImage(&src_gau_R);
	cvReleaseImage(&src_gau_G);
	cvReleaseImage(&src_gau_B);
	cvReleaseImage(&src_gau);

	return 0;
}

int connectgaps(unsigned char * img, int imageW, int imageH) {
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
	

int checkneighbors(unsigned char * canny, unsigned char * weakEdges, unsigned char *visited, int i, int j, int imageW, int imageH) {
	//must have borders of weakedges=0

	int k, w;

	for (k = -1; k <= 1; k++) {
		for (w = -1; w <= 1; w++) {
			if (k != 0 || w != 0) {
				if (weakEdges[(i + k) + j + w] == 1 && visited[(i + k)*imageW + j + w] == 0) {
					visited[(i + k)*imageW + j + w] = 1;
					canny[(i + k)*imageW + j + w] = 1;
					checkneighbors(canny, weakEdges, visited, i + k, j + w, imageW, imageH);
				}
			}
		}
	}

	/*if ((weakEdges[i*imageW + j + 1] == 1) && (j + 1 < imageW) && (visited[i*imageW+j+1]==0)) { //PREPEI NA ELEGKSW TA ORIA APO PRIN
		visited[i*imageW + j + 1] = 1;
		canny[i*imageW + j + 1] = 1;
		checkneighbors(canny, weakEdges, visited, i, j + 1, imageW, imageH);
	}
	else if ((weakEdges[(i + 1)*imageW + j] == 1) && (i + 1 < imageH) && (visited[(i+1)*imageW + j] == 0)) {
		visited[(i+1)*imageW + j] = 1;
		canny[(i + 1)*imageW + j] = 1;
		checkneighbors(canny, weakEdges, visited, i + 1, j, imageW, imageH);
	}
	else if ((weakEdges[(i + 1)*imageW + j + 1] == 1) && (i + 1 < imageH) && (j + 1< imageW) && (visited[(i+1)*imageW + j + 1] == 0)) {
		visited[(i+1)*imageW + j + 1] = 1;
		canny[(i + 1)*imageW + j + 1] = 1;
		checkneighbors(canny, weakEdges, visited, i + 1, j + 1, imageW, imageH);
	}
	else if (weakEdges[(i - 1)*imageW + j] == 1 && (i - 1) >= 0 && (visited[(i-1)*imageW + j] == 0)) {
		visited[(i-1)*imageW + j] = 1;
		canny[(i - 1)*imageW + j] = 1;
		checkneighbors(canny, weakEdges, visited, i - 1, j + 1,imageW,imageH);
	}
	else if (weakEdges[(i - 1)*imageW + j - 1] == 1 && i - 1 >= 0 && j + 1 < imageW && (visited[(i - 1)*imageW + j - 1] == 0)) {
		visited[(i-1)*imageW + j - 1] = 1;
		canny[(i - 1)*imageW + j - 1] = 1;
		checkneighbors(canny, weakEdges, visited, i - 1, j - 1, imageW, imageH);
	}
	else if (weakEdges[i*imageW + j - 1] == 1 && j - 1 < imageW && (visited[i*imageW + j - 1] == 0)) {
		visited[i*imageW + j - 1] = 1;
		canny[i*imageW + j - 1] = 1;
		checkneighbors(canny, weakEdges, visited, i, j - 1, imageW, imageH);
	}
	else if (weakEdges[(i + 1)*imageW + j - 1] == 1 && (i + 1 < imageH) && (j - 1 >= 0) && (visited[(i + 1)*imageW + j - 1] == 0)){
		visited[(i+1)*imageW + j - 1] = 1;
		canny[(i+1)*imageW + j - 1] = 1;
		checkneighbors(canny, weakEdges, visited, i + 1, j - 1, imageW, imageH);
	} 
	else if (weakEdges[(i - 1)*imageW + j + 1] == 1 && (i - 1 >= 0) && (j + 1 < imageW) && (visited[(i - 1)*imageW + j + 1] == 0)) {
		visited[(i - 1)*imageW + j + 1] = 1;
		canny[(i - 1)*imageW + j + 1] = 1;
		checkneighbors(canny, weakEdges, visited, i - 1, j + 1, imageW, imageH);
	}*/
	return 1;
}