#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>

#include <cv.h>						//vision processing
#include <highgui.h>

#include <Mmsystem.h>
#include <conio.h>
#include <windows.h>
#include <winbase.h>
#include <stdio.h>

#include "VisionProcessingCode.h"
#include "SocketComm.h"				//feature detection (matlab) and action (robot)


#define MAX_NUM_POSITIONS 10000

#define HIST_SIZE 10
#define BUFSIZE 1024

#define NUM_SURFACES 12
#define NUM_TRIALS_PER_OBJECT 120
#define NUM_MARK_ATTEMPTS 10
#define MARK_DETECTED_THRESHOLD 10  

#define SPIRAL 0
#define LINE 1
#define DOTS 2
#define HORIZONTAL 3
#define VERTICAL 4

#define NUM_ROTATIONS 10
#define ROTATION_INCREMENT 2
#define NUM_SCALES 10
#define SCALE_INCREMENT 5

/*
 Code for the Science Channel Demo of :
 Sahai. Griffith. and Stoytchev. ``Interactive Learning of Writing Instruments and Writable Surfaces by a Robot.''
 RSS Manipulation Workshop. 2009
 */


char * rootname = "C:/Learning2Write";
IplImage * hist[HIST_SIZE];

void GetNextImageName(char * filename, int trial, int image)
{
	sprintf(filename, "%s/i%.3d/rgb%.6d.jpg", rootname, trial, image);
}

void UpdateHist(IplImage * mask)
{
	for(int i=0; i<HIST_SIZE-1; i++)
	{
		cvCopy(hist[i+1], hist[i], NULL);
	}
	cvCopy(mask, hist[HIST_SIZE-1], NULL);
}

void MergeHistory(IplImage  * merged)
{
	static IplImage * sum;
	if(sum == NULL) sum = cvCreateImage(cvSize(merged->width, merged->height), IPL_DEPTH_8U, 1);
	
	cvZero(sum);
	for(int i=0; i<HIST_SIZE; i++)
	{
		cvAdd(sum, hist[i], sum, NULL);
	}
	cvThreshold(sum, merged, 1, 255, CV_THRESH_BINARY);
}

void GetBiggestBlob(IplImage * blobs, IplImage * biggestBlob)
{
	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq * maxContour = 0;
	CvSeq* contour = 0;
	int biggestBlobArea = -1;
	CvContourScanner traverse = NULL;

	traverse = cvStartFindContours(blobs, storage, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	contour = cvFindNextContour(traverse);

	while(contour != NULL)
	{
		int contourarea = (int) abs(cvContourArea(contour,CV_WHOLE_SEQ));
		if(contourarea > biggestBlobArea)
		{
			biggestBlobArea = contourarea;
			maxContour = contour;
		}
		contour = cvFindNextContour(traverse);
	}
	
	cvDrawContours(biggestBlob, maxContour, cvScalar(100), cvScalar(100), -1, CV_FILLED, 8);

	contour = cvEndFindContours(&traverse);
	cvReleaseMemStorage(&storage);
} 

void RemoveSmallBlobs(IplImage * src, IplImage * dst, int minArea)
{
	IplImage * temp = cvCloneImage(src);
	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq* contour = 0;
	CvContourScanner traverse = NULL;

	traverse = cvStartFindContours(src, storage, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	contour = cvFindNextContour(traverse);

	while(contour != NULL)
	{
		int contourarea = (int) abs(cvContourArea(contour,CV_WHOLE_SEQ));
		if(contourarea > minArea)
		{
			cvDrawContours(dst, contour, cvScalar(255, 0, 0, 0), cvScalar(255, 0, 0, 0), -1, CV_FILLED, 8);
		}
		contour = cvFindNextContour(traverse);
	}

	contour = cvEndFindContours(&traverse);
	cvReleaseMemStorage(&storage);

	cvAnd(temp, dst, dst, 0);
	cvReleaseImage(&temp);
} 

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

//write the string in 'str' to the file specified
void appendToFile(char filename[], char str[])
{
     FILE *fp = NULL;
     
     //if the file does not exist yet, it is created and opened for writing.
     //if the file exists, the contents of 'str' are appended to the end
     fp = fopen(filename, "a");
     fputs(str, fp);
     fclose(fp);
}

int GetNumberOfImages(char * trialdata)
{
	char * token;
	int i;
	for(token = strtok(trialdata, ","), i=0; token != NULL; token = strtok(NULL, ","), i++)
	{
		if(i==4)
		{
			return (int) strtol(token, NULL, 10);
		}
	}
}

int GetTypeOfMark(char * trialdata)
{
	char * token;
	int i;
	for(token = strtok(trialdata, ","), i=0; token != NULL; token = strtok(NULL, ","), i++)
	{
		if(i==3)
		{
			if(strcmp("draw_a_spiral", token)==0) return SPIRAL;
			else if(strcmp("draw_a_straight_line", token)==0) return LINE;
			else if(strcmp("make_dots", token)==0) return DOTS;
			else if(strcmp("scribble_horizontally", token)==0) return HORIZONTAL;
			else if(strcmp("scribble_vertically", token)==0) return VERTICAL;
			else printf("no match found.\n");
		}
	}
	return -1;
}

double getSum(IplImage * img)
{
	if(img->nChannels == 1)
	{
		double sum=0.0;
		BwImage _img(img);
		for(int i=0; i<img->height; i++)
		{
			for(int j=0; j<img->width; j++ )
			{
				sum+= _img[i][j];
			}
		}
		return sum;
	}
	else
	{
		double sumb=0.0, sumg=0.0, sumr=0.0;
		RgbImage _img(img);
		for(int i=0; i<img->height; i++)
		{
			for(int j=0; j<img->width; j++ )
			{
				sumb+= _img[i][j].b;
				sumg+= _img[i][j].g;
				sumr+= _img[i][j].r;
			}
		}
		return sumb+sumg+sumr;
	}
}

void CreateTemplate(char * filename, IplImage * src, double y_offset, int border_offset)
{
	double cartesian_position[MAX_NUM_POSITIONS][3];
	int pixel_position[MAX_NUM_POSITIONS][3];
	int length;
	double cMAX[3];
	double cMIN[3];
	cMAX[0] = cMAX[1] = cMAX[2] = (double) INT_MIN;
	cMIN[0] = cMIN[1] = cMIN[2] = (double) INT_MAX;

	double pMAX[3];
	double pMIN[3];
	pMAX[0] = src->width-border_offset;
	pMAX[2] = src->height-border_offset;
	pMIN[0] = pMIN[2] = border_offset;

	FILE * fp = fopen(filename, "r");
	char BUFFER[BUFSIZE];
	for(int i=0; fgets(BUFFER, sizeof(BUFFER), fp) != NULL && i<MAX_NUM_POSITIONS; i++)
	{
		char * token = strtok(BUFFER, ",");
		for(int j=0; token != NULL; j++)
		{
			if(j==1) //get the y_min position before the other values (because the other values depend on the y_min)
			{
				cartesian_position[i][j] = strtod(token, NULL);
				if(cartesian_position[i][j] < cMIN[j]) cMIN[j] = cartesian_position[i][j];
				if(cartesian_position[i][j] > cMAX[j]) cMAX[j] = cartesian_position[i][j];
			}
			token = strtok(NULL, ",");
		}
	}
	fclose(fp);

	//load the trajectory data and save the max/min ranges
	fp = fopen(filename, "r");
	for(int i=0; fgets(BUFFER, sizeof(BUFFER), fp) != NULL && i<MAX_NUM_POSITIONS; i++)
	{
		double tMIN[3];
		double tMAX[3];
		char * token = strtok(BUFFER, ",");
		for(int j=0; token != NULL; j++)
		{
			cartesian_position[i][j] = strtod(token, NULL);
			tMIN[j] = (cartesian_position[i][j] < cMIN[j])  ?  cartesian_position[i][j] : cMIN[j];
			tMAX[j] = (cartesian_position[i][j] > cMAX[j])  ?  cartesian_position[i][j] : cMAX[j];
			token = strtok(NULL, ",");
		}

		//only save the x and z range when the y value is close to the table
		if(cartesian_position[i][1] > cMIN[1] + y_offset) 
		{
			cMIN[0] = tMIN[0];
			cMIN[2] = tMIN[2];
			cMAX[0] = tMAX[0];
			cMAX[2] = tMAX[2];
		}
		length = i;
	}
	fclose(fp);

	if(length == MAX_NUM_POSITIONS-1) printf("The number of positions exceeded __%d__\n", MAX_NUM_POSITIONS);
	else printf("There were __%d__ positions in the file.\n", length);
	
	//compute the transformation values.
	double stretch_transformation[3];
	stretch_transformation[0] = (pMAX[0]-pMIN[0])/(cMAX[0]-cMIN[0]);
	stretch_transformation[2] = (pMAX[2]-pMIN[2])/(cMAX[2]-cMIN[2]);

	//use the smaller multiplier
	double multiplier = ( stretch_transformation[0] > stretch_transformation[2] )  ?  stretch_transformation[2]  :  stretch_transformation[0];

	int shift[3];
	shift[0] = 0.5 * (pMAX[0]-pMIN[0]) * (1 - multiplier/stretch_transformation[0]);
	shift[2] = 0.5 * (pMAX[2]-pMIN[2]) * (1 - multiplier/stretch_transformation[2]);

	printf("shift: (%d, %d)\n", shift[0], shift[2]);

	for(int i=0; i<length; i++)
	{
		for(int j=0; j<3; j++)
		{
			pixel_position[i][j] = shift[j] + multiplier * (cartesian_position[i][j]-cMIN[j]);
		}
	}

	BwImage _img(src);
	for(int i=0; i<length; i++)
	{
		int pX = pMAX[0] - pixel_position[i][0];
		int pY = pMAX[2] - pixel_position[i][2];
		for(int j=0; j<3;j++)
		{
			if(j == 1) continue;
			if(pX > pMAX[j]) pX = pMAX[j];
			if(pY < pMIN[j]) pY = pMIN[j];
		}

		if(cartesian_position[i][1] > cMIN[1] + y_offset) 
			_img[pX][pY] = 255;
	} 

}

void RotateAndScaleImage(IplImage * src, IplImage * dest, double scale, double angle)
{
	float m[6];
    CvMat M = cvMat(2, 3, CV_32F, m);
	m[0] = (float)(scale*cos(-angle*2*CV_PI/180.));
    m[1] = (float)(scale*sin(-angle*2*CV_PI/180.));
    m[3] = -m[1];
    m[4] = m[0];
	m[2] = src->width*0.5f;  
	m[5] = src->height*0.5f; 
    cvGetQuadrangleSubPix( src, dest, &M);
}

void RotateImage(IplImage * src, IplImage * dest, double angle)
{
	RotateAndScaleImage(src, dest, 1, angle);
}

void EraseBorder(IplImage * src)
{
	if(src->nChannels == 1)
	{
		BwImage _img(src);

		for(int i=0; i<src->width; i++)
		{
			_img[i][0] = 0;
			_img[i][src->height-1] = 0;
		}
		for(int i=0; i<src->height; i++)
		{
			_img[0][i] = 0;
			_img[src->width-1][i] = 0;
		}
	}
	else
	{
		RgbImage _img(src);
		for(int i=0; i<src->width; i++)
		{
			_img[i][0].r = _img[i][0].b = _img[i][0].g = 0;
			_img[i][src->height-1].r = _img[i][src->height-1].b = _img[i][src->height-1].g = 0;
		}
		for(int i=0; i<src->height; i++)
		{
			_img[0][i].r = _img[0][i].b = _img[0][i].g = 0;
			_img[src->width-1][i].r = _img[src->width-1][i].b = _img[src->width-1][i].g = 0;
		}
	}
}

IplImage * CreateLearningToWriteTemplate(int mark)
{
	CvRect rect = cvRect(0, 0, 250, 250);
	IplImage * marked_image = cvCreateImage(cvSize(rect.width, rect.height), IPL_DEPTH_8U, 1);
	IplImage * disp_image = cvCloneImage(marked_image);

	int border_offset = 25;
	char filename[50] = "";
	double angle;
	double y_offset;
	
	switch(mark)
	{
		case SPIRAL:
			strcat(filename, "behaviors/draw_a_spiral_CARTESIAN.csv");
			angle = -28;
			y_offset = 0.03;
			break;
		case LINE:
			strcat(filename,"behaviors/draw_a_straight_line_CARTESIAN.csv");
			angle = -14;
			y_offset = 0.02;
			break;
		case DOTS:
			strcat(filename,"behaviors/make_dots_CARTESIAN.csv");
			angle = -15;
			y_offset = 0.03;
			break;
		case HORIZONTAL:
			strcat(filename,"behaviors/scribble_horizontally_CARTESIAN.csv");
			angle = -18;
			y_offset = 0.03;
			break;
		case VERTICAL:
			strcat(filename,"behaviors/scribble_vertically_CARTESIAN.csv");
			angle = -10;
			y_offset = 0.03;
			break;
	}

	CreateTemplate(filename, marked_image, y_offset, border_offset);

	morphImage(marked_image, marked_image, MORPH_DILATE, 2);

	RotateImage(marked_image, disp_image, angle);

	return cvCloneImage(disp_image);
}


void CreateIdealMark()
{
	CvRect rect = cvRect(0, 0, 250, 250);
	IplImage * marked_image = cvCreateImage(cvSize(rect.width, rect.height), IPL_DEPTH_8U, 1);
	IplImage * disp_image = cvCloneImage(marked_image);
	IplImage * res_image = cvCreateImage(cvSize(50, 50), IPL_DEPTH_8U, 1);
	char savename[100];

	int border_offset = 25;
	int mark = VERTICAL;

	char filename[50] = "";
	double angle;
	double y_offset;
	
	switch(mark)
	{
		case SPIRAL:
			strcpy(savename, "draw_a_spiral.jpg");
			strcat(filename, "behaviors/draw_a_spiral_CARTESIAN.csv");
			angle = -28;
			y_offset = 0.03;
			break;
		case LINE:
			strcpy(savename, "draw_a_straight_line.jpg");
			strcat(filename,"behaviors/draw_a_straight_line_CARTESIAN.csv");
			angle = -14;
			y_offset = 0.02;
			break;
		case DOTS:
			strcpy(savename, "make_dots.jpg");
			strcat(filename,"behaviors/make_dots_CARTESIAN.csv");
			angle = -15;
			y_offset = 0.03;
			break;
		case HORIZONTAL:
			strcpy(savename, "scribble_horizontally.jpg");
			strcat(filename,"behaviors/scribble_horizontally_CARTESIAN.csv");
			angle = -18;
			y_offset = 0.03;
			break;
		case VERTICAL:
			strcpy(savename, "scribble_vertically.jpg");
			strcat(filename,"behaviors/scribble_vertically_CARTESIAN.csv");
			angle = -10;
			y_offset = 0.03;
			break;
	}

	CreateTemplate(filename, marked_image, y_offset, border_offset);

	morphImage(marked_image, marked_image, MORPH_DILATE, 2);

	RotateImage(marked_image, disp_image, angle);

	cvSaveImage(savename, disp_image);

	cvNamedWindow("Ideal Mark", 1);
	cvShowImage("Ideal Mark", res_image);
	cvWaitKey(0);
	cvDestroyWindow("Ideal Mark");
}

void rotatetest()
{
    IplImage* src = cvLoadImage("shane.jpg", 1);    
    IplImage* dst = cvCloneImage( src );

    double delta = 0.01;
    double angle = 0;
    int opt = 0;   // 1 rotate & zoom
                   // 0:  rotate only
    double factor;
    cvNamedWindow("src", 1);
    cvShowImage("src", src);

    for(;;)
    {
        float m[6];
        CvMat M = cvMat(2, 3, CV_32F, m);
        int w = src->width;
        int h = src->height;

        if(opt)  
            factor = (cos(angle*CV_PI/180.) + 1.05) * 2;
        else 
            factor = 1;
        m[0] = (float)(factor*cos(-angle*2*CV_PI/180.));
        m[1] = (float)(factor*sin(-angle*2*CV_PI/180.));
        m[3] = -m[1];
        m[4] = m[0];
        m[2] = w*0.5f;  
        m[5] = h*0.5f;  
        
        cvGetQuadrangleSubPix( src, dst, &M);
        cvNamedWindow("dst", 1);
        cvShowImage("dst", dst);
        if( cvWaitKey(1) == 27 )
            break;
		angle = (angle + delta > 360)? angle + delta - 360: angle + delta;
    }     
    return;
}


void FormatAndPrintResults(char * formattedName, int cur_iteration, int mark_detected)
{
	static double freq_marks[NUM_SURFACES];
	static double mark_detected_counter = 0.0;

	mark_detected_counter += mark_detected;

	if(cur_iteration!= 0 && (cur_iteration+1)%NUM_MARK_ATTEMPTS == 0)
	{
		if((cur_iteration+1)%NUM_TRIALS_PER_OBJECT == 0)
		{
			char formatted[500];
			char temp[500];
	
			sprintf(formatted, "%lf, ", freq_marks[0]);
			for(int l=1; l<NUM_SURFACES-1; l++)
			{
				sprintf(temp, "%lf, ", freq_marks[l]);
				strcat(formatted, temp);
			}
			sprintf(temp, "%lf\n", mark_detected_counter/10);
			strcat(formatted, temp);

			printf("formatted: %s\n", formatted);

			FILE * fp_res;
	
			if(cur_iteration+1 == NUM_MARK_ATTEMPTS*NUM_SURFACES)
			{
				fp_res = fopen(formattedName, "w");
			}
			else
			{
				fp_res = fopen(formattedName, "a");
			}

			if(fp_res == NULL)
			{
				printf("Could not open %s for writing.\n", formattedName);
				return;
			}

			if((cur_iteration+1)==NUM_TRIALS_PER_OBJECT) fputs(formatted, fp_res);
			else fputs(formatted, fp_res);

			fclose(fp_res);
		}
		else
		{
			freq_marks[(((cur_iteration+1)/NUM_MARK_ATTEMPTS)-1)%NUM_SURFACES] = mark_detected_counter/10;
			//getchar();
			//printf("i=%d, for=%d\n, count", i, freq_marks[((i+1)/NUM_MARK_ATTEMPTS)-1%NUM_SURFACES], count);
		}

		mark_detected_counter = 0.0;
	}
}

void SetCaptureParams(CvCapture * capture, int width, int height)
{
  //set properties
  cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, width);
  cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, height); 
}

/**
 *  Get the capture structure from the camera at the specified deviceNum.
 *  
 *  @return capture structure representing the chosen device
 *
 */
CvCapture * CaptureFromCamera(){
  //set capture to capture from the webcam
  CvCapture * capture = cvCreateCameraCapture(-1);
  if( !capture ){
    printf("Could not initialize capturing.\n");
	//end the current program instead of exiting the program
    exit(EXIT_FAILURE);
  }

  cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, 640);
  cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, 480);
  
  return capture;
}

void ScienceChannel()
{
	IplImage * c_img;
	IplImage * g_img;
	IplImage * c_diff;
	IplImage * c_last;
	IplImage * c_first;
	IplImage * g_first;
	IplImage * g_last;
	IplImage * g_diff;
	IplImage * filtered_small;
	IplImage * save_image;

	CvFont font;
	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 2.0, 2.0, 0, 2, CV_AA);
	cvNamedWindow("Mark Detection", CV_WINDOW_AUTOSIZE);
	cvMoveWindow("Mark Detection", 0, 0);
	cvNamedWindow("Robot Field of View", CV_WINDOW_AUTOSIZE);
	cvMoveWindow("Robot Field of View", 250, 0);

	const int static_param_FILTER_METHOD = 6;

	int param_FIRST_i =0;
	int param_MARK_AREA_WIDTH = 640;	//250
	int param_MARK_AREA_HEIGHT = 480;	//200
	int param_MARK_X_OFFSET = 0;		//137
	int param_MARK_Y_OFFSET = 0;       //148
	int param_MARK_HALF_X_OFFSET = 0;  //68
	int param_MARK_HALF_Y_OFFSET = 0; //74

	double mark_detected = 0;
	double total = 0.0;

	char image_filename[500];
	char result[100];
	char BUFFER[BUFSIZE];
    char curline[BUFSIZE];

	CvCapture * capture = CaptureFromCamera();
	int lastsize = 0;
	for(int i=0;;i++)
	{
		char input_char = '\0';
		printf("Waiting for FIRST frame. Enter __c__ to continue. Enter __q__ to quit\n");
		while(input_char != 'c')
		{
			scanf("%c", &input_char);
		}

		c_img = cvLoadImage("Picture 10.jpg", 1);
		if(!c_img)
		{
			printf("Could not get image!\n");
			exit(EXIT_FAILURE);
		}

		if(lastsize != c_img->width)
		{
			if(i != param_FIRST_i)
			{
				cvReleaseImage(&g_img);
				cvReleaseImage(&c_last);
				cvReleaseImage(&c_first);
				cvReleaseImage(&c_diff);
				cvReleaseImage(&g_first);
				cvReleaseImage(&g_last);
				cvReleaseImage(&g_diff);
				cvReleaseImage(&filtered_small);
			}

			lastsize = c_img->width;
			if(c_img->width == 640)
			{
				c_first = cvCreateImage(cvSize(param_MARK_AREA_WIDTH, param_MARK_AREA_HEIGHT), c_img->depth, 3);
				c_last = cvCreateImage(cvSize(param_MARK_AREA_WIDTH, param_MARK_AREA_HEIGHT), c_img->depth, 3);
				c_diff = cvCreateImage(cvSize(param_MARK_AREA_WIDTH, param_MARK_AREA_HEIGHT), c_img->depth, 3);
				g_img = cvCreateImage(cvSize(param_MARK_AREA_WIDTH, param_MARK_AREA_HEIGHT), c_img->depth, 1);
				g_first = cvCreateImage(cvSize(param_MARK_AREA_WIDTH, param_MARK_AREA_HEIGHT), c_img->depth, 1);
				g_last = cvCreateImage(cvSize(param_MARK_AREA_WIDTH, param_MARK_AREA_HEIGHT), c_img->depth, 1);
				g_diff = cvCreateImage(cvSize(param_MARK_AREA_WIDTH, param_MARK_AREA_HEIGHT), c_img->depth, 1);
				filtered_small = cvCreateImage(cvSize(param_MARK_AREA_WIDTH, param_MARK_AREA_HEIGHT), c_img->depth, 1);
			}
			else
			{
				c_first = cvCreateImage(cvSize(param_MARK_AREA_WIDTH/2, param_MARK_AREA_HEIGHT/2), c_img->depth, 3);
				c_last = cvCreateImage(cvSize(param_MARK_AREA_WIDTH/2, param_MARK_AREA_HEIGHT/2), c_img->depth, 3);
				c_diff = cvCreateImage(cvSize(param_MARK_AREA_WIDTH/2, param_MARK_AREA_HEIGHT/2), c_img->depth, 3);
				g_img = cvCreateImage(cvSize(param_MARK_AREA_WIDTH/2, param_MARK_AREA_HEIGHT/2), c_img->depth, 1);
				g_first = cvCreateImage(cvSize(param_MARK_AREA_WIDTH/2, param_MARK_AREA_HEIGHT/2), c_img->depth, 1);
				g_last = cvCreateImage(cvSize(param_MARK_AREA_WIDTH/2, param_MARK_AREA_HEIGHT/2), c_img->depth, 1);
				g_diff = cvCreateImage(cvSize(param_MARK_AREA_WIDTH/2, param_MARK_AREA_HEIGHT/2), c_img->depth, 1);
				filtered_small = cvCreateImage(cvSize(param_MARK_AREA_WIDTH/2, param_MARK_AREA_HEIGHT/2), c_img->depth, 1);
			}
		}

		if(c_img->width == 640) cvSetImageROI(c_img, cvRect(param_MARK_X_OFFSET, param_MARK_Y_OFFSET, param_MARK_AREA_WIDTH, param_MARK_AREA_HEIGHT));
		else cvSetImageROI(c_img, cvRect(param_MARK_HALF_X_OFFSET, param_MARK_HALF_Y_OFFSET, param_MARK_AREA_WIDTH/2, param_MARK_AREA_HEIGHT/2));

		cvCopy(c_img, c_first, NULL);
		cvCvtColor(c_img, g_img, CV_BGR2GRAY);
		cvCopy(g_img, g_first, NULL);
		cvResetImageROI(c_img);

		input_char = '\0';
		printf("Waiting for LAST frame. Enter __c__ to continue.\n");
		while(input_char != 'c')
		{
			if(CheckRxBuffer(fileno(stdin)))
			{
				scanf("%c", &input_char);
			}
			c_img = cvQueryFrame(capture);
			if(!c_img)
			{
				printf("Could not get image at __%s__!\n", image_filename);
				exit(EXIT_FAILURE);
			}
			cvShowImage("Robot Field Of View", c_img);
			cvWaitKey(30);
		}

		if(!c_img)
		{
			printf("Could not get image at __%s__!\n", image_filename);
			exit(EXIT_FAILURE);
		}

		c_img = cvLoadImage("Picture 11.jpg", 1);

		if(c_img->width == 640) cvSetImageROI(c_img, cvRect(param_MARK_X_OFFSET, param_MARK_Y_OFFSET, param_MARK_AREA_WIDTH, param_MARK_AREA_HEIGHT));
		else cvSetImageROI(c_img, cvRect(param_MARK_HALF_X_OFFSET, param_MARK_HALF_Y_OFFSET, param_MARK_AREA_WIDTH/2, param_MARK_AREA_HEIGHT/2));

		cvCopy(c_img, c_last, NULL);
		cvCvtColor(c_img, g_img, CV_BGR2GRAY);
		cvCopy(g_img, g_last, NULL);
		cvResetImageROI(c_img);

		double sum = 0.0;
		//update all of these to reflect the image width differences.
		switch(static_param_FILTER_METHOD)
		{
		case 0:
			cvAbsDiff(c_last, c_first, c_diff);
			cvThreshold(c_diff, c_diff, 18, 0, CV_THRESH_TOZERO);  //set to 0 the pixels that changed by less than 7%
			sum = getSum(c_diff);  //determine the amount of change
			mark_detected = (sum > 500)  ?  1 : 0; //a mark is detected when the image changed by more than 500
			save_image = c_diff;
			break;
		case 1:
			cvAbsDiff(g_last, g_first, g_diff);
			cvThreshold(g_diff, g_diff, 18, 0, CV_THRESH_TOZERO);  //set to 0 the pixels that changed by less than 7%
			sum = getSum(g_diff);  //determine the amount of change
			mark_detected = (sum > 500)  ?  1 : 0; //a mark is detected when the image changed by more than 500  
			save_image = g_diff;
			break;
		case 2:
			cvAbsDiff(c_last, c_first, c_diff);
			cvThreshold(c_diff, c_diff, 18, 0, CV_THRESH_TOZERO);  //set to 0 the pixels that changed by less than 7%
			cvThreshold(c_diff, c_diff, 0, 1, CV_THRESH_BINARY);   //set to 1 the rest of the pixels
			sum = getSum(c_diff);  //count the number of pixels that changed 
			mark_detected = (sum > 500 || (c_diff->width == 320 && sum > 125))  ?  1 : 0; //a mark was detected if over 500 pixels changed
			save_image = c_diff;
			break;
		case 3:
			cvAbsDiff(g_last, g_first, g_diff);
			cvThreshold(g_diff, g_diff, 18, 0, CV_THRESH_TOZERO);  //set to 0 the pixels that changed by less than 7%
			cvThreshold(g_diff, g_diff, 0, 1, CV_THRESH_BINARY);   //set to 1 the rest of the pixels
			sum = getSum(g_diff);  //count the number of pixels that changed 
			mark_detected = (sum > 500 || (g_diff->width == 320 && sum > 125))  ?  1 : 0; //a mark was detected if over 500 pixels changed 
			save_image = g_diff;
			break;
		case 4:
			//the images used for figures (these look cleaner compared to the method using color)
			cvAbsDiff(g_first, g_last, g_diff);
			cvThreshold(g_diff, g_diff, 18, 0, CV_THRESH_TOZERO);
			cvThreshold(g_diff, g_diff, 0, 255, CV_THRESH_BINARY);
			save_image = g_diff;
			break;
		case 5:
			cvAbsDiff(c_last, c_first, c_diff);
			cvThreshold(c_diff, c_diff, 18, 0, CV_THRESH_TOZERO);  //set to 0 the pixels that changed by less than 7%
			cvCvtColor(c_diff, g_diff, CV_BGR2GRAY);
			cvThreshold(g_diff, g_diff, 0, 255, CV_THRESH_BINARY);
			cvZero(filtered_small);
			RemoveSmallBlobs(g_diff, filtered_small, 20);
			sum = getSum(filtered_small);  //determine the amount of change
			mark_detected = (sum > 500)  ?  1 : 0; //a mark is detected when the image changed by more than 500
			save_image = filtered_small;
			break;
		case 6:
			cvAbsDiff(c_last, c_first, c_diff);
			cvThreshold(c_diff, c_diff, 18, 0, CV_THRESH_TOZERO);  //set to 0 the pixels that changed by less than 7%
			cvCvtColor(c_diff, g_diff, CV_BGR2GRAY);
			cvThreshold(g_diff, g_diff, 0, 255, CV_THRESH_BINARY);
			cvCopy(g_diff, filtered_small, 0);

			break;
		default:
			break;
		}

		int scale = 4;
		IplImage * displayed = cvCreateImage(cvSize(filtered_small->width*scale, filtered_small->height*scale), c_img->depth, c_img->nChannels);
		IplImage * filtered_color = cvCreateImage(cvSize(filtered_small->width, filtered_small->height), filtered_small->depth, c_img->nChannels);
		
		cvCvtColor(filtered_small, filtered_color, CV_GRAY2BGR);
		cvResize(filtered_color, displayed, 1);
		cvFlip(displayed, displayed, 0);

		//result is in mark_detected (double) 1 when a mark, 0 when not
		if(mark_detected > 0)
		{
			cvPutText(displayed, "mark detected!", cvPoint(15, 40), &font, CV_RGB(255, 0, 0)); //put text in the image
		}
		else
		{
			cvPutText(displayed, "no mark...", cvPoint(55, 40), &font, CV_RGB(255, 0, 0)); //put text in the image
		}

		
		cvRectangle(c_img, cvPoint(param_MARK_HALF_X_OFFSET, param_MARK_HALF_Y_OFFSET), cvPoint(param_MARK_HALF_X_OFFSET+param_MARK_AREA_WIDTH/2, param_MARK_HALF_Y_OFFSET+param_MARK_AREA_HEIGHT/2), CV_RGB(255, 255, 255), 1, 8, 0);

		cvShowImage("Mark Detection", displayed);
		cvWaitKey(0);
		cvReleaseImage(&displayed);
		cvReleaseImage(&filtered_color);
	}

	cvDestroyWindow("Mark Detection");

	return;
}

void ShowRobotView()
{
	IplImage * img;
	CvCapture * capture = CaptureFromCamera();
	cvNamedWindow("robot view");
	for(;;)
	{
		img = cvQueryFrame(capture);
		cvShowImage("robot view", img);
		cvWaitKey(30);
	}
	cvDestroyWindow("robot view");
	cvReleaseCapture(&capture);
}	

int main()
{	
	const int static_param_PROGRAM = 4;
	const int static_param_FILTER_METHOD = 5;
	const int static_param_SAVE_IMAGE = 1;
	const int static_param_FORMAT_RESULTS = 0;
	const int static_param_DEBUG = 0;
	
	int param_FIRST_i = 0;				
	int param_NUM_TRIALS = 1440;		
	int param_MARK_AREA_WIDTH = 250;	
	int param_MARK_AREA_HEIGHT = 200;	
	int param_MARK_X_OFFSET = 137;		
	int param_MARK_Y_OFFSET = 148;       
	int param_MARK_HALF_X_OFFSET = 68;  
	int param_MARK_HALF_Y_OFFSET = 74; 
	
	switch(static_param_PROGRAM)
	{
	case 0:
		//use all the default parameter values
		break;
	case 1: //get one large image
		int image_num;
		printf("Enter the trial number.\n");
		scanf("%d", &image_num);
		if(image_num < 0 || image_num > param_NUM_TRIALS)
		{
			printf("Could not process trial %d. Choose a value between %d and %d.\n", image_num, param_FIRST_i, param_NUM_TRIALS);
			exit(1);
		}
		param_FIRST_i = image_num;
		param_NUM_TRIALS = param_FIRST_i + 1;  //set the parameters such that only the selected image is processed.

		param_MARK_AREA_WIDTH = 640;
		param_MARK_AREA_HEIGHT = 480;
		param_MARK_X_OFFSET = 0;
		param_MARK_Y_OFFSET = 0;
		param_MARK_HALF_X_OFFSET = 0;
		param_MARK_HALF_Y_OFFSET = 0;
		break;
	case 2:CreateIdealMark();
		exit(1);
	case 3: rotatetest();
		exit(1);
	case 4: ScienceChannel();
		exit(1);
	case 5: ShowRobotView();
		exit(1);
	default:
		break;
	}

	IplImage * c_img;
	IplImage * g_img;
	IplImage * c_diff;
	IplImage * c_last;
	IplImage * c_first;
	IplImage * g_first;
	IplImage * g_last;
	IplImage * g_diff;
	IplImage * filtered_small;
	IplImage * save_image;

	double mark_detected = 0;
	double total = 0.0;

	char image_filename[500];

	char * formattedName = "formatted results.csv";
	char * resultsName = "mark_detection.csv";
	char result[100];

	char * trialData = "C:/Learning2Write/ExperimentData.txt";
	char BUFFER[BUFSIZE];
    char curline[BUFSIZE];
	
	FILE * fp = NULL;
	fp = fopen(trialData, "r");
	if(fp == NULL)
	{
		printf("%s does not exist. Could not read.\n", trialData);
		return 0;
	}
	
	for(int j=0; j<param_FIRST_i; j++)
	{
		fgets(BUFFER, BUFSIZE, fp);
	}

	int lastsize = 0;
	for(int i=param_FIRST_i; i<param_NUM_TRIALS; i++)
	{
		GetNextImageName(image_filename, i, 1);
		c_img = cvLoadImage(image_filename, 1);
		if(!c_img)
		{
			printf("Could not get image at __%s__!\n", image_filename);
			exit(EXIT_FAILURE);
		}

		if(lastsize != c_img->width)
		{
			if(i != param_FIRST_i)
			{
				cvReleaseImage(&g_img);
				cvReleaseImage(&c_last);
				cvReleaseImage(&c_first);
				cvReleaseImage(&c_diff);
				cvReleaseImage(&g_first);
				cvReleaseImage(&g_last);
				cvReleaseImage(&g_diff);
				cvReleaseImage(&filtered_small);
			}

			lastsize = c_img->width;
			if(c_img->width == 640)
			{
				c_first = cvCreateImage(cvSize(param_MARK_AREA_WIDTH, param_MARK_AREA_HEIGHT), c_img->depth, 3);
				c_last = cvCreateImage(cvSize(param_MARK_AREA_WIDTH, param_MARK_AREA_HEIGHT), c_img->depth, 3);
				c_diff = cvCreateImage(cvSize(param_MARK_AREA_WIDTH, param_MARK_AREA_HEIGHT), c_img->depth, 3);
				g_img = cvCreateImage(cvSize(param_MARK_AREA_WIDTH, param_MARK_AREA_HEIGHT), c_img->depth, 1);
				g_first = cvCreateImage(cvSize(param_MARK_AREA_WIDTH, param_MARK_AREA_HEIGHT), c_img->depth, 1);
				g_last = cvCreateImage(cvSize(param_MARK_AREA_WIDTH, param_MARK_AREA_HEIGHT), c_img->depth, 1);
				g_diff = cvCreateImage(cvSize(param_MARK_AREA_WIDTH, param_MARK_AREA_HEIGHT), c_img->depth, 1);
				filtered_small = cvCreateImage(cvSize(param_MARK_AREA_WIDTH, param_MARK_AREA_HEIGHT), c_img->depth, 1);
			}
			else
			{
				c_first = cvCreateImage(cvSize(param_MARK_AREA_WIDTH/2, param_MARK_AREA_HEIGHT/2), c_img->depth, 3);
				c_last = cvCreateImage(cvSize(param_MARK_AREA_WIDTH/2, param_MARK_AREA_HEIGHT/2), c_img->depth, 3);
				c_diff = cvCreateImage(cvSize(param_MARK_AREA_WIDTH/2, param_MARK_AREA_HEIGHT/2), c_img->depth, 3);
				g_img = cvCreateImage(cvSize(param_MARK_AREA_WIDTH/2, param_MARK_AREA_HEIGHT/2), c_img->depth, 1);
				g_first = cvCreateImage(cvSize(param_MARK_AREA_WIDTH/2, param_MARK_AREA_HEIGHT/2), c_img->depth, 1);
				g_last = cvCreateImage(cvSize(param_MARK_AREA_WIDTH/2, param_MARK_AREA_HEIGHT/2), c_img->depth, 1);
				g_diff = cvCreateImage(cvSize(param_MARK_AREA_WIDTH/2, param_MARK_AREA_HEIGHT/2), c_img->depth, 1);
				filtered_small = cvCreateImage(cvSize(param_MARK_AREA_WIDTH/2, param_MARK_AREA_HEIGHT/2), c_img->depth, 1);
			}
		}

		if(c_img->width == 640) cvSetImageROI(c_img, cvRect(param_MARK_X_OFFSET, param_MARK_Y_OFFSET, param_MARK_AREA_WIDTH, param_MARK_AREA_HEIGHT));
		else cvSetImageROI(c_img, cvRect(param_MARK_HALF_X_OFFSET, param_MARK_HALF_Y_OFFSET, param_MARK_AREA_WIDTH/2, param_MARK_AREA_HEIGHT/2));

		cvCopy(c_img, c_first, NULL);
		cvCvtColor(c_img, g_img, CV_BGR2GRAY);
		cvCopy(g_img, g_first, NULL);
		cvResetImageROI(c_img);
		cvReleaseImage(&c_img);

		if(fgets(BUFFER, BUFSIZE, fp) == NULL)
		{
			printf("result is NULL\n");
			exit(EXIT_FAILURE);
		}

		GetNextImageName(image_filename, i, GetNumberOfImages(BUFFER)  );
		c_img = cvLoadImage(image_filename, 1);
		if(!c_img)
		{
			printf("Could not get image at __%s__!\n", image_filename);
			exit(EXIT_FAILURE);
		}

		if(c_img->width == 640) cvSetImageROI(c_img, cvRect(param_MARK_X_OFFSET, param_MARK_Y_OFFSET, param_MARK_AREA_WIDTH, param_MARK_AREA_HEIGHT));
		else cvSetImageROI(c_img, cvRect(param_MARK_HALF_X_OFFSET, param_MARK_HALF_Y_OFFSET, param_MARK_AREA_WIDTH/2, param_MARK_AREA_HEIGHT/2));

		cvCopy(c_img, c_last, NULL);
		cvCvtColor(c_img, g_img, CV_BGR2GRAY);
		cvCopy(g_img, g_last, NULL);
		cvResetImageROI(c_img);
		cvReleaseImage(&c_img);

		double sum = 0.0;
		//update all of these to reflect the image width differences.
		switch(static_param_FILTER_METHOD)
		{
		case 0:
			cvAbsDiff(c_last, c_first, c_diff);
			cvThreshold(c_diff, c_diff, 18, 0, CV_THRESH_TOZERO);  //set to 0 the pixels that changed by less than 7%
			sum = getSum(c_diff);  //determine the amount of change
			mark_detected = (sum > 500)  ?  1 : 0; //a mark is detected when the image changed by more than 500
			save_image = c_diff;
			break;
		case 1:
			cvAbsDiff(g_last, g_first, g_diff);
			cvThreshold(g_diff, g_diff, 18, 0, CV_THRESH_TOZERO);  //set to 0 the pixels that changed by less than 7%
			sum = getSum(g_diff);  //determine the amount of change
			mark_detected = (sum > 500)  ?  1 : 0; //a mark is detected when the image changed by more than 500  
			save_image = g_diff;
			break;
		case 2:
			cvAbsDiff(c_last, c_first, c_diff);
			cvThreshold(c_diff, c_diff, 18, 0, CV_THRESH_TOZERO);  //set to 0 the pixels that changed by less than 7%
			cvThreshold(c_diff, c_diff, 0, 1, CV_THRESH_BINARY);   //set to 1 the rest of the pixels
			sum = getSum(c_diff);  //count the number of pixels that changed 
			mark_detected = (sum > 500 || (c_diff->width == 320 && sum > 125))  ?  1 : 0; //a mark was detected if over 500 pixels changed
			save_image = c_diff;
			break;
		case 3:
			cvAbsDiff(g_last, g_first, g_diff);
			cvThreshold(g_diff, g_diff, 18, 0, CV_THRESH_TOZERO);  //set to 0 the pixels that changed by less than 7%
			cvThreshold(g_diff, g_diff, 0, 1, CV_THRESH_BINARY);   //set to 1 the rest of the pixels
			sum = getSum(g_diff);  //count the number of pixels that changed 
			mark_detected = (sum > 500 || (g_diff->width == 320 && sum > 125))  ?  1 : 0; //a mark was detected if over 500 pixels changed 
			save_image = g_diff;
			break;
		case 4:
			//the images used for figures (these look cleaner compared to the method using color)
			cvAbsDiff(g_first, g_last, g_diff);
			cvThreshold(g_diff, g_diff, 18, 0, CV_THRESH_TOZERO);
			cvThreshold(g_diff, g_diff, 0, 255, CV_THRESH_BINARY);
			save_image = g_diff;
			break;
		case 5:
			cvAbsDiff(c_last, c_first, c_diff);
			cvThreshold(c_diff, c_diff, 18, 0, CV_THRESH_TOZERO);  //set to 0 the pixels that changed by less than 7%
			cvCvtColor(c_diff, g_diff, CV_BGR2GRAY);
			cvThreshold(g_diff, g_diff, 0, 255, CV_THRESH_BINARY);
			cvZero(filtered_small);
			RemoveSmallBlobs(g_diff, filtered_small, 20);
			sum = getSum(filtered_small);  //determine the amount of change
			mark_detected = (sum > 500)  ?  1 : 0; //a mark is detected when the image changed by more than 500
			save_image = filtered_small;
			break;
		case 6:
			break;
		default:
			break;
		}

		sprintf(result, "%f\n", mark_detected);
		if(i==param_FIRST_i) writeToFile(resultsName, result);
		else appendToFile(resultsName, result);

		if(static_param_DEBUG)
		{
			cvNamedWindow("debug", 1);
			cvShowImage("debug", save_image);
			cvWaitKey(0);
			cvDestroyWindow("debug");

			if(mark_detected == 1) printf("mark detected\n");
		}

		if(static_param_FORMAT_RESULTS)
		{
			FormatAndPrintResults(formattedName, i, (int) mark_detected);
		}
		
		if(static_param_SAVE_IMAGE)
		{
			char savename[100];
			sprintf(savename, "C:/Learning2Write/i%.3d/detected_mark.jpg", i);
			cvSaveImage(savename, save_image);
		}

		printf("Analyzed trial=%d\n", i);
	}

	fclose(fp);
	return 0;
}