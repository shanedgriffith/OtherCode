/*
 An implementation of :
 X. Liu and D. Wang. ``A Spectral Histogram Model for Texton Modeling and Texture Discrimination.''
 Vision Research. v. 42, no. 23, pp. 2617-2634. 2002
 
 @author: Shane Griffith.
 
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>

#include <tdvCameraInterface.h>		//Primary camera include file
#include "DepthCamera.h"
#include <cv.h>
#include <highgui.h>

#include <Mmsystem.h>
#include <conio.h>
#include <windows.h>
#include <winbase.h>

#include "stdafx.h"
#include "DataWriter.h"
#include "SocketComm.h"

#include "cvgabor.h"

#include <stdio.h>

#define INTENSITY					0
#define LOCAL_DIFFERENCE_Dx			1
#define LOCAL_DIFFERENCE_Dy			2
#define LOCAL_DIFFERENCE_Dxx		3
#define LOCAL_DIFFERENCE_Dyy		4
#define LAPLACIAN_OF_GAUSSIAN_1		5  //uses sqrt(2)/2
#define LAPLACIAN_OF_GAUSSIAN_2		6  //uses 2 (others could be 1 and 4)
#define GABOR_1						7
#define GABOR_2						8
#define GABOR_3						9
#define GABOR_4						10
#define GABOR_5						11
#define GABOR_6						12
#define GABOR_7						13
#define GABOR_8						14

#define KERNEL_WIDTH	5
#define KERNEL_HEIGHT	5

#define NUM_FILTERS					13
#define NUM_BINS					10

#define BUFSIZE						5000

typedef struct
{
	double hist[NUM_FILTERS][NUM_BINS];
} SPECTRAL_HISTOGRAM;

/**Compute a sub-band image through linear convolution.
 *
 */
void conv_img(CvMat * kernel, IplImage *src, IplImage *dst)
{
	double ve, re,im;
	CvMat *mat = cvCreateMat(src->width, src->height, CV_32FC1);

	/****Convert src image to floating point ****/
	for (int i = 0; i < src->width; i++)
	{
		for (int j = 0; j < src->height; j++)
		{
			ve = CV_IMAGE_ELEM(src, uchar, i, j); //j, i);
			CV_MAT_ELEM(*mat, float, i, j) = (float)ve;
		}
	}

	cvFilter2D( (CvMat*)mat, (CvMat*)mat, (CvMat*)kernel, cvPoint( (kernel->width-1)/2, (kernel->height-1)/2));

	/****Convert result to desired output****/
	if (dst->depth == IPL_DEPTH_8U)
	{
		cvNormalize((CvMat*)mat, (CvMat*)mat, 0, 255, CV_MINMAX);
		for (int i = 0; i < mat->rows; i++)
		{
			for (int j = 0; j < mat->cols; j++)
			{
				ve = CV_MAT_ELEM(*mat, float, j, i); //i, j);
				CV_IMAGE_ELEM(dst, uchar, j, i) = (uchar)cvRound(ve);
			}
		}
	}
	else if (dst->depth == IPL_DEPTH_32F)
	{
		for (int i = 0; i < mat->rows; i++)
		{
			for (int j = 0; j < mat->cols; j++)
			{
				ve = cvGetReal2D((CvMat*)mat, j, i);
				cvSetReal2D( dst, j, i, ve );
			}
		}
	}

	cvReleaseMat(&mat);
}

IplImage * GetImage(CvMat * kernel)
{ 
	double ve;
    CvScalar S;
    CvSize size = cvGetSize( kernel );
    int rows = size.height;
    int cols = size.width;
    IplImage* pImage = cvCreateImage( size, IPL_DEPTH_32F, 1 );
    IplImage *newimage = cvCreateImage(size, IPL_DEPTH_8U, 1 );
    CvMat* k_temp = cvCreateMat(kernel->width, kernel->width, CV_32FC1);

	cvCopy( (CvMat*)kernel, (CvMat*)k_temp, NULL );
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			ve = cvGetReal2D((CvMat*)k_temp, i, j);
			cvSetReal2D( (IplImage*)pImage, i, j, ve); //j, i, ve );
		}
	}
   
    cvNormalize((IplImage*)pImage, (IplImage*)pImage, 0, 255, CV_MINMAX, NULL );

    cvConvertScaleAbs( (IplImage*)pImage, (IplImage*)newimage, 1, 0 );

    cvReleaseMat(&k_temp);
    cvReleaseImage(&pImage);

    return newimage;
}

void IntensityFilter(CvMat * kernel, int width, int height)
{

	for(int i=0; i<kernel->height; i++)
	{
		for(int j=0; j<kernel->width; j++)
		{
			CV_MAT_ELEM(*kernel, float, i, j) = 0.0;
		}
	}
	CV_MAT_ELEM(*kernel, float, height/2, width/2) = 2.0;
}

void LocalDifferenceFilter(CvMat * kernel, int width, int height)
{
	//CvMat * ldf_kernel = cvCreateMat(width, height, CV_32FC1);

	for(int i=0; i<kernel->height; i++)
	{
		for(int j=0; j<kernel->width; j++)
		{
			CV_MAT_ELEM(*kernel, float, i, j) = 0.0;
		}
	}

	//return ldf_kernel;
}

void GaborKernel(CvMat * kernel, int width, int height, double Phi, int iNu, double Sigma, double F)
{
	int x, y;
	double dReal;
	double dTemp1, dTemp2;

	double Kmax = PI/2;
	double K = Kmax / pow(F, (double)iNu);


	/**************************** Gabor Function ****************************/ 
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			x = j-(width-1)/2;
			y = i-(height-1)/2;
			dTemp1 = (pow(K,2)/pow(Sigma,2))*exp(-(pow((double)x,2)+pow((double)y,2))*pow(K,2)/(2*pow(Sigma,2)));
			dTemp2 = cos(K*cos(Phi)*x + K*sin(Phi)*y) - exp(-(pow(Sigma,2)/2));
			dReal = dTemp1*dTemp2;
			
			cvSetReal2D((CvMat*)kernel, i, j, dReal);
		}
	}

	//return gab1_kernel;
}

void LoG(CvMat * kernel, int width, int height, double Variance)
{
	int x, y;
	double dTemp;
	double T_squared = pow(Variance*sqrt(2.0), 2);

	/**************************** Gabor Function ****************************/ 
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			x = j-(width-1)/2;
			y = i-(height-1)/2;
			dTemp = (pow((double)x,2) + pow((double)y,2) - T_squared)*exp(-(pow((double)x,2) + pow((double)y,2))/T_squared);
			
			cvSetReal2D((CvMat*)kernel, i, j, dTemp);
		}
	}
}

void GetKernel(CvMat * kernel, int knum, int width, int height)
{
	//CvMat * kernel;
	double Phi;			//orientation
	int iNu;			//scale
	double Sigma;		//sigma value
	double F;			//frequency
	double Variance;

	switch(knum)
	{
	case INTENSITY:
		IntensityFilter(kernel, width, height);
		return;
		break;
	case LOCAL_DIFFERENCE_Dx:  //Gradient Filter Dx
		LocalDifferenceFilter(kernel, width, height);
		CV_MAT_ELEM(*kernel, float, width/2-1, height/2) = 0.0;
		CV_MAT_ELEM(*kernel, float, width/2, height/2) = -1.0;
		CV_MAT_ELEM(*kernel, float, width/2+1, height/2) = 1.0;
		return;
		break;
	case LOCAL_DIFFERENCE_Dy:  //Gradient Filter Dy
		LocalDifferenceFilter(kernel, width, height);
		CV_MAT_ELEM(*kernel, float, width/2, height/2-1) = 0.0;
		CV_MAT_ELEM(*kernel, float, width/2, height/2) = -1.0;
		CV_MAT_ELEM(*kernel, float, width/2, height/2+1) = 1.0;
		return;
		break;
	case LOCAL_DIFFERENCE_Dxx: //Gradient Filter Dxx 
		LocalDifferenceFilter(kernel, width, height);
		CV_MAT_ELEM(*kernel, float, width/2-1, height/2) = -1.0;
		CV_MAT_ELEM(*kernel, float, width/2, height/2) = 2.0;
		CV_MAT_ELEM(*kernel, float, width/2+1, height/2) = -1.0;
		return;
		break;
	case LOCAL_DIFFERENCE_Dyy:  //Gradient Filter Dyy
		LocalDifferenceFilter(kernel, width, height);
		CV_MAT_ELEM(*kernel, float, width/2, height/2-1) = -1.0;
		CV_MAT_ELEM(*kernel, float, width/2, height/2) = 2.0;
		CV_MAT_ELEM(*kernel, float, width/2, height/2+1) = -1.0;
		return;
		break;
	case LAPLACIAN_OF_GAUSSIAN_1:
		Variance = 0.5;
		LoG(kernel, width, height, Variance);
		return;
		break;
	case LAPLACIAN_OF_GAUSSIAN_2:
		Variance = 4/sqrt(2.0);
		LoG(kernel, width, height, Variance);
		return;
		break;
	case GABOR_1:
		Phi = PI;	//orientation
		iNu = 1;		//scale
		Sigma = 2*PI;	//sigma value
		F = sqrt(2.0);	//frequency
		GaborKernel(kernel, width, height, Phi, iNu, Sigma, F);
		return;
		break;
	case GABOR_2:
		Phi = PI/2;		//orientation
		iNu = 1;		//scale
		Sigma = 2*PI;	//sigma value
		F = sqrt(2.0);	//frequency
		GaborKernel(kernel, width, height, Phi, iNu, Sigma, F);
		return;
		break;	
	case GABOR_3:
		Phi = PI/4;		//orientation
		iNu = 1;		//scale
		Sigma = 2*PI;	//sigma value
		F = sqrt(2.0);	//frequency
		GaborKernel(kernel, width, height, Phi, iNu, Sigma, F);
		return;
		break;
	case GABOR_4:
		Phi = -PI/4;		//orientation
		iNu = 1;		//scale
		Sigma = 2*PI;	//sigma value
		F = sqrt(2.0);	//frequency
		GaborKernel(kernel, width, height, Phi, iNu, Sigma, F);
		return;
		break;
	case GABOR_5:
		Phi = PI;		//orientation
		iNu = 1;		//scale
		Sigma = 2*PI;	//sigma value
		F = sqrt(2.0)/2;	//frequency
		GaborKernel(kernel, width, height, Phi, iNu, Sigma, F);
		return;
		break;
	case GABOR_6:
		Phi = PI/3;		//orientation
		iNu = 1;		//scale
		Sigma = 2*PI;	//sigma value
		F = sqrt(2.0);	//frequency
		GaborKernel(kernel, width, height, Phi, iNu, Sigma, F);
		return;
		break;
	case GABOR_7:
		Phi = PI/4;		//orientation
		iNu = 3;		//scale
		Sigma = 2*PI;	//sigma value
		F = sqrt(2.0)/2;	//frequency
		GaborKernel(kernel, width, height, Phi, iNu, Sigma, F);
		return;
		break;
	//case GABOR_8:
	//	Phi = -PI/4;		//orientation
	//	iNu = 3;		//scale
	//	Sigma = 2*PI;	//sigma value
	//	F = 2*sqrt(2.0);	//frequency
	//	GaborKernel(kernel, width, height, Phi, iNu, Sigma, F);
	//	return;
	//	break;
	default:
		printf("Bad request (%d).\n", knum);
	}
}

void GetFilters(CvMat * kernels[NUM_FILTERS])
{
	int i;
	
	for(i=0; i<NUM_FILTERS; i++)
	{
		kernels[i] = cvCreateMat(KERNEL_HEIGHT, KERNEL_WIDTH, CV_32FC1);
		GetKernel(kernels[i], i, KERNEL_WIDTH, KERNEL_HEIGHT);
	}
}

/** Compute the histogram (marginal distribution) of the subband image.
 * Hist (z) = 1/|num pixels in image| * SUM{v, delta*(z-subband_img[v])}
 * --where the SUM,v is just over all the pixels in the image.
 * --where the intensity filter, delta, is a function that captures the intensity value at a given pixel.
 * --where z denotes the zth bin of the histogram???
 */
void Histogram(IplImage * subband, double hist[], int numbins)
{
	//this algorithm finds the min and max value, computes the bin values, and normalizes to unit mean and zero standard deviation

	double min = 1.0*INT_MAX;
	double max = 1.0*INT_MIN;
	double ve;
	double width;
	int binno;
	double low, high;
	int i, j;

	if(subband->depth != IPL_DEPTH_32F)
	{
		printf("This operation requires the subband image.\n");
		return;
		//exit(EXIT_FAILURE);
	}

	//compute the min and max
	for(i=0; i<subband->height; i++)
	{
		for(j=0; j<subband->width; j++)
		{
			ve = cvGetReal2D(subband, i, j);
			if(ve < min) min = ve;
			if(ve > max) max = ve;
		}
	}

	//determine the width of a bin
	width = (max-min)/numbins;

	//initialize the histogram
	for(i=0; i<NUM_BINS; i++)
	{
		hist[i] = 0;
	}

	//add the min to everything, then divide by the width to get the bin number.
	for(i=0; i<subband->height; i++)
	{
		for(j=0; j<subband->width; j++)
		{
			ve = cvGetReal2D(subband, i, j);
			binno = ((ve - min)/width); 
			binno = (binno == NUM_BINS)? binno-1: binno;
			hist[binno]++;
		}
	}


	//normalize to between 0 and 1
	for(i=0; i<NUM_BINS; i++)
	{
		hist[i] /= subband->height*subband->width;
	}
	
}

void GetSpectralHistogram(SPECTRAL_HISTOGRAM * sp, IplImage * img, CvMat * kernels[])
{
	IplImage* subband = cvCreateImage( cvSize(img->width, img->height), IPL_DEPTH_32F, 1 );  // 8U, 1); //

	int i;
	//cvNamedWindow("asdf", 1);
	for(i=0; i<NUM_FILTERS; i++)
	{
		if(kernels[i] == NULL)
		{
			printf("Found a NULL kernel. That ain't going to work.\n");
			exit(EXIT_FAILURE);
		}

		conv_img(kernels[i], img, subband);
		
		Histogram(subband, sp->hist[i], NUM_BINS);
	}

	cvReleaseImage(&subband);
}

IplImage * GetResponseImage(SPECTRAL_HISTOGRAM * sp, IplImage * img, CvMat * kernels[])
{
	IplImage* subband = cvCreateImage( cvSize(img->width, img->height), IPL_DEPTH_8U, 1);
	IplImage * response = cvCreateImage( cvSize(img->width, img->height), IPL_DEPTH_8U, 1);
	cvZero(response);
	for(int i=0; i<NUM_FILTERS; i++)
	{
		if(kernels[i] == NULL)
		{
			printf("Found a NULL kernel. That ain't going to work.\n");
			exit(EXIT_FAILURE);
		}

		conv_img(kernels[i], img, subband);
		cvNamedWindow("a", 1);
		cvShowImage("a", subband);
		cvWaitKey(0);
		cvDestroyWindow("a");
		cvCopy(subband, response, 0);
	}

	cvReleaseImage(&subband);
	return response;
}

double CHI_Squared(SPECTRAL_HISTOGRAM a, SPECTRAL_HISTOGRAM b)
{
	double sim=0.0;
	int i, j;

	for(i=0; i<NUM_FILTERS; i++)
	{
		for(j=0; j<NUM_BINS; j++)
		{
			sim += pow((a.hist[i][j] - b.hist[i][j]), 2)/(a.hist[i][j] + b.hist[i][j]);
		}
	}

	sim /= NUM_FILTERS;

	return sim;
}

int FindMostSimilar(SPECTRAL_HISTOGRAM novel, SPECTRAL_HISTOGRAM surface_categories[], int num_categories)
{
	//min CHI-squared()
	int i;
	int min;
	double sim;

	for(i=0; i<num_categories; i++)
	{
		sim = CHI_Squared(novel, surface_categories[i]);
		if(sim < min) min = i;
	}

	return min;
}

//write the string in 'str' to the file specified
void writeToFile(char filename[], char str[])
{
     FILE *fp = NULL;
     
     //if the file does not exist yet, it is created and opened for writing.
     //if the file exists, the contents of the current file are erased and 
     //overwritten with the contents of 'str'
     fp = fopen(filename, "w");
     fputs(str, fp);
     fclose(fp);
}

void appendToFile(char filename[], char str[])
{
     FILE *fp = NULL;
     
     //if the file does not exist yet, it is created and opened for writing.
     //if the file exists, the contents of 'str' are appended to the end
     fp = fopen(filename, "a");
     fputs(str, fp);
     fclose(fp);
}

/* Save the list of spectral histograms 
 * -save the number of surfaces to cluster on the first line
 * -save the number of filters on the second line
 * -save the histograms on the following lines
 */
void SaveComputedSpectralHistograms(char * filename, int num_surfaces, SPECTRAL_HISTOGRAM interacted_surfaces[])
{
	char BUFFER[BUFSIZE];
	char NUM_BUF[500];
	int i, j, k;
	
	sprintf(BUFFER, "");
	writeToFile(filename, BUFFER);

	for(i=0; i<num_surfaces; i++)
	{
		for(j=0; j<NUM_FILTERS; j++)
		{
			for(k=0; k<NUM_BINS; k++)
			{
				if(j== NUM_FILTERS-1 && k == NUM_BINS-1)
				{
					sprintf(NUM_BUF, "%lf\n", interacted_surfaces[i].hist[j][k]);
				}
				else
				{
					sprintf(NUM_BUF, "%lf,", interacted_surfaces[i].hist[j][k]);
				}
				
				if(j==0 && k==0)
				{
					strcpy(BUFFER, NUM_BUF);
				}
				else
				{
					strcat(BUFFER, NUM_BUF);
				}
			}
		}
		appendToFile(filename, BUFFER);
		BUFFER[0] = '\0';
	}
}

/* Load a file that specifies on first line 1: num clusters, following: first column, a number; proceeding columns, spectral histogram for that surface
 * -load the number to determine the number of spectral histograms there are.
 * -create as many spectral histograms as there are clusters. 
 * -As each spectral histogram is loaded, add the hist values the sum of each spectral histogram
 * -keep a count of the number of surfaces in that spectral histogram
 * -when finished with iterating the file, compute the average of each spectral histogram (no re-marginalization is required)
 * -return this array of spectral histograms.
 */
SPECTRAL_HISTOGRAM * GetLearnedSurfaceCategories(char * filename)
{
	char BUFFER[BUFSIZE];
	int line_iter=0, token_iter=0;
	int i, j, k;
	int cur_category=0;
	int num_clusters;
	char * token;
	SPECTRAL_HISTOGRAM * surface_categories;
	int * count_categories;

	FILE * fp = NULL;
	fp = fopen(filename, "r");
	if(fp == NULL)
	{
		printf("%s does not exist. Could not modify.", filename);
		return NULL;
	}

	while(fgets(BUFFER, BUFSIZE, fp) != NULL)
	{
		if(line_iter == 0)
		{
			num_clusters = (int) strtol(BUFFER, NULL, 10);
			surface_categories = (SPECTRAL_HISTOGRAM *) malloc(num_clusters*sizeof(SPECTRAL_HISTOGRAM));
			count_categories = (int *) malloc(num_clusters*sizeof(int));
			
			//initialize to zeros
			for(i=0; i<num_clusters; i++)
			{
				count_categories[i] = 0;

				for(j=0; j<NUM_FILTERS; j++)
					for(k=0; k<NUM_BINS; k++)
						surface_categories[i].hist[j][k] = 0.0;
			}

		}
		else
		{
			for(token = strtok(BUFFER, ","), token_iter = 0; token; token = strtok(NULL, ","), token_iter++)
			{
				if(token_iter == 0)
				{
					cur_category = strtol(token, NULL, 10);
					count_categories[cur_category]++;
				}
				else
				{
					surface_categories[cur_category].hist[(token_iter-1)/NUM_FILTERS][(token_iter-1)%NUM_FILTERS] += strtod(token, NULL);
				}
			}
		}

		line_iter++;
	}

	//divide each by the total number in that cluster
	for(i=0; i<num_clusters; i++)
		for(j=0; j<NUM_FILTERS; j++)
			for(k=0; k<NUM_BINS; k++)
				surface_categories[i].hist[j][k] /= count_categories[i];

	return surface_categories;
}

void GenerateSpectralHistograms(char * fileLoc, int num_surfaces, SPECTRAL_HISTOGRAM interacted_surfaces[])
{
	IplImage * img;		
	IplImage * gray;
	char BUFFER[BUFSIZE];
	int i;
	CvMat * kernels[NUM_FILTERS];


	GetFilters(kernels);

	for(i=0; i<num_surfaces; i++)
	{
		sprintf(BUFFER, "%s_%.2d.jpg", fileLoc, i+1);
		img = cvLoadImage(BUFFER, 1);
		if(!img)
		{
			printf("File at __%s__ could not be loaded!\n", BUFFER);
		}
		printf("File at __%s__ loaded!\n", BUFFER);
		
		if(i==0) gray = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_8U, 1);
		cvCvtColor(img, gray, CV_BGR2GRAY);
		
		GetSpectralHistogram(&interacted_surfaces[i], gray, kernels);
	}

	if(gray != NULL) cvReleaseImage(&gray);
}

void ShowFilterResponses(char * fileLoc, int num_surfaces, SPECTRAL_HISTOGRAM interacted_surfaces[])
{
	IplImage * img;		
	IplImage * gray;
	char BUFFER[BUFSIZE];
	int i;
	CvMat * kernels[NUM_FILTERS];


	GetFilters(kernels);

	for(i=0; i<num_surfaces; i++)
	{
		sprintf(BUFFER, "%s_%.2d.jpg", fileLoc, i+1);
		img = cvLoadImage(BUFFER, 1);
		if(!img)
		{
			printf("File at __%s__ could not be loaded!\n", BUFFER);
		}
		printf("File at __%s__ loaded!\n", BUFFER);
		
		if(i==0) gray = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_8U, 1);
		cvCvtColor(img, gray, CV_BGR2GRAY);
		
		//GetSpectralHistogram(&interacted_surfaces[i], gray, kernels);
		IplImage * response = GetResponseImage(&interacted_surfaces[i], gray, kernels);
		cvReleaseImage(&response);
	}

	if(gray != NULL) cvReleaseImage(&gray);
}

void CreateSparseCodingImages(char * fileLoc, int num_surfaces, int size)
{
	IplImage * img;		
	IplImage * gray;
	IplImage * res = cvCreateImage(cvSize(size, size), IPL_DEPTH_8U, 1);
	char BUFFER[BUFSIZE];
	char str[250];

	for(int i=0; i<num_surfaces; i++)
	{
		sprintf(BUFFER, "%s_%.2d.jpg", fileLoc, i+1);
		img = cvLoadImage(BUFFER, 1);
		if(!img)
		{
			printf("File at __%s__ could not be loaded!\n", BUFFER);
		}
		printf("File at __%s__ loaded!\n", BUFFER);
		
		if(i==0) gray = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_8U, 1);
		cvCvtColor(img, gray, CV_BGR2GRAY);

		cvResize(gray, res, 1);
		sprintf(str, "D:/school/Vision Research/Code/Sparse Coding/EpiRob Code/cropped_depth_s30/img%d.jpg", i+1);
		cvSaveImage(str, res);
	}

	if(gray != NULL) cvReleaseImage(&gray);
}

void GetFilterPictures()
{
	const int num_rows = 2;
	IplImage * kimg;
	IplImage * res = cvCreateImage(cvSize((NUM_FILTERS/num_rows)*(KERNEL_WIDTH+5), (num_rows+1)*(KERNEL_HEIGHT+5)), IPL_DEPTH_8U, 1);

	cvZero(res);
	for(int i=0; i<NUM_FILTERS; i++)
	{
		CvMat * kernel = cvCreateMat(KERNEL_HEIGHT, KERNEL_WIDTH, CV_32FC1);;
		GetKernel(kernel, i, KERNEL_WIDTH, KERNEL_HEIGHT);
		if(kernel == NULL) continue;
		kimg = GetImage(kernel);

		CvRect rec = cvRect((i%(NUM_FILTERS/num_rows))*(KERNEL_WIDTH+5), (i/(NUM_FILTERS/num_rows))*(KERNEL_HEIGHT+5), KERNEL_WIDTH, KERNEL_HEIGHT);
		printf("x,y, wid, hei(%d,%d,%d,%d)\n", rec.x, rec.y, rec.width, rec.height);
		cvSetImageROI(res, rec);
		cvCopy(kimg, res, NULL);
		cvResetImageROI(res);
		
		cvReleaseImage(&kimg);
		cvReleaseMat(&kernel);
	}
	cvSaveImage("FilterPictures.jpg", res);
}


int main(int argc, char *argv[])
{
	const int program = 3;
	const int size = 30;
	const int num_surfaces = 12;


	SPECTRAL_HISTOGRAM interacted_surfaces[num_surfaces];


	switch(program)
	{
	case 0:
		GenerateSpectralHistograms("C:/Temp/small_surfaces/surface", num_surfaces, interacted_surfaces);
		SaveComputedSpectralHistograms("C:/Learning/spectral_histograms.csv", num_surfaces, interacted_surfaces);
		break;
	case 1:
		ShowFilterResponses("C:/Temp/small_surfaces/surface", num_surfaces, interacted_surfaces);
		break;
	case 2:
		GetFilterPictures();
		break;
	case 3:
		CreateSparseCodingImages("C:/Temp/small_surfaces/surface", num_surfaces, size);
		break;
	default:
		break;
	}
	
	return 0;
}








