#include "stdafx.h"
#include<iostream>
#include <opencv2/opencv.hpp>
#include "opencv2\highgui\highgui.hpp"
#include <windows.h>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

#define window_name "My Window"
#define ORIG_IMG_NAME "picture005.jpg"
#define MARKED_IMG_NAME "picture005.jpg"

using namespace std;
using namespace cv;

void logo_detector(Rect rect);
void edge_detector(Mat a);
void draw_graph(Mat a);

vector<Point2f> center_points;
bool found_logo;

int main(void)
{
	cout<<"Video Recording... Press 'ESC' to stop.\n";

	VideoCapture cap(0);
	Mat frameObj;
 
	//Read the video from the webcam
	while((cap.read(frameObj))!=NULL)
	{

		imwrite(MARKED_IMG_NAME,frameObj);
		
		edge_detector(frameObj);
		
		//imshow(window_name,display);

		char terminate=cvWaitKey(10);
		if(terminate==27)
		{
			cout << "ESC key pressed" << endl;
			break;
		}
	}
	draw_graph(frameObj);
	return 0;
}

Mat frame2;
int display_dst( int delay )
{
  imshow( "window_name2", frame2 );
  int c = waitKey ( delay );
  if( c >= 0 ) { return -1; }
  return 0;
}

void draw_graph(Mat frame)
{
	//frame2=frame.clone();
	//frame.setTo(cv::Scalar(255,255,255));
	if(center_points.size()==0)
		return;

	for(int i=0;i<(center_points.size()-1);i++)
	{
		line(frame,center_points[i],center_points[i+1],Scalar(0,0,255),2,8);
		if(center_points[i].x<center_points[i+1].x && center_points[i].y<center_points[i+1].y)
		{
			line(frame,Point(center_points[i+1].x-10,center_points[i+1].y-5),Point(center_points[i+1].x,center_points[i+1].y),Scalar(0,0,0),2,8);
			line(frame,Point(center_points[i+1].x-10,center_points[i+1].y+5),Point(center_points[i+1].x,center_points[i+1].y),Scalar(0,0,0),2,8);
		}
		else if(center_points[i].x<center_points[i+1].x && center_points[i].y>center_points[i+1].y)
		{
			line(frame,Point(center_points[i+1].x+10,center_points[i+1].y-5),Point(center_points[i+1].x,center_points[i+1].y),Scalar(0,0,0),2,8);
			line(frame,Point(center_points[i+1].x+10,center_points[i+1].y+5),Point(center_points[i+1].x,center_points[i+1].y),Scalar(0,0,0),2,8);
		}
		else if(center_points[i].x>=center_points[i+1].x && center_points[i].y<center_points[i+1].y)
		{
			line(frame,Point(center_points[i+1].x+10,center_points[i+1].y-5),Point(center_points[i+1].x,center_points[i+1].y),Scalar(0,0,0),2,8);
			line(frame,Point(center_points[i+1].x+10,center_points[i+1].y+5),Point(center_points[i+1].x,center_points[i+1].y),Scalar(0,0,0),2,8);
		}
		else
		{
			line(frame,Point(center_points[i+1].x+10,center_points[i+1].y-5),Point(center_points[i+1].x,center_points[i+1].y),Scalar(0,0,0),2,8);
			line(frame,Point(center_points[i+1].x+10,center_points[i+1].y+5),Point(center_points[i+1].x,center_points[i+1].y),Scalar(0,0,0),2,8);
		}
	}
	imshow("my_final_path",frame);
	/*int j=3;
	for(int i=0;i<(center_points.size()-1);i++)
	{
		
		line(frame2,center_points[i],center_points[i+1],Scalar(255-((i*20)%255),255-((i*20)%255),255-((i*20)%255)),2,8);
		if(j!=0)
		{
			j--;
			continue;
		}
		display_dst(1500);
	}*/
	waitKey(0);	
}

void logo_detector(Rect orig_rect)
{
	Mat dst;
	Mat src=imread("cropped_pic.jpg");
	GaussianBlur( src, src, Size(5,5), 2, 2 );
	cvtColor(src,src,CV_BGR2HSV);
	inRange(src, cv::Scalar(89,8,103), cv::Scalar(180, 256, 256), dst);
	

	int erosion_type=2,erosion_size=1;
	Mat erosion_dst;
	Mat element = getStructuringElement( erosion_type, Size( 2*erosion_size + 1, 2*erosion_size+1 ),Point( erosion_size, erosion_size ) );
	erode( dst, erosion_dst, element );

	erosion_size=15;
	element = getStructuringElement( erosion_type, Size( 2*erosion_size + 1, 2*erosion_size+1 ),Point( erosion_size, erosion_size ) );
	dilate( erosion_dst, erosion_dst, element );
	//imshow("window",erosion_dst);
	
	vector<Vec4i> hierarchy;
	vector<vector<Point> > contours;
	
	Canny( erosion_dst, erosion_dst, 80, 80*2, 3 );
	findContours(erosion_dst,contours,hierarchy,CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE);
	Mat orig=imread(MARKED_IMG_NAME);
	if(contours.size()==0)
	{
		found_logo=false;
		return;
	}
	int largest_area=0;
	int largest_contour_index=0;
	Rect bounding_rect;
	
	vector<Point2f>center( contours.size() );
	vector<float>radius( contours.size() );
	vector<vector<Point> > contours_poly( contours.size());
	vector<Rect> boundRect( contours.size());

	int j;
	for( int i = 0; i< contours.size(); i++ )
	{
		double a=contourArea( contours[i],false);
		if(a>largest_area){
			approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
			boundRect[i] = boundingRect( Mat(contours_poly[i]) );
			minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
			largest_area=a;
			largest_contour_index=i;                
			bounding_rect=boundingRect(contours[i]); 
			j=i;
		}
	}

	Mat drawing = Mat::zeros( erosion_dst.size(), CV_8UC3 );
	Scalar color( 0,255,255);
	drawContours( drawing, contours,largest_contour_index, color, 2, 8, hierarchy );
	rectangle(orig, Point(orig_rect.x+bounding_rect.x,orig_rect.y+bounding_rect.y), Point(orig_rect.x+bounding_rect.x+bounding_rect.width,orig_rect.y+bounding_rect.y+bounding_rect.height), Scalar(255,0,0),2, 8,0);
	found_logo=true;
	imwrite("picture005.jpg",orig);

}

void edge_detector(Mat img)
{
	namedWindow( window_name, WINDOW_AUTOSIZE );
	Mat src=imread(MARKED_IMG_NAME);
	
	//Using smoothing function reduce noise in the image
	GaussianBlur( src, src, Size(7,7), 2, 2 );
	Mat dst;
	cvtColor(src,src,CV_BGR2HSV);
	inRange(src, cv::Scalar(0,173,22), cv::Scalar(24, 255, 255), dst);

	int erosion_type=2,erosion_size=8;
	Mat erosion_dst;
	Mat element = getStructuringElement( erosion_type, Size( 2*erosion_size + 1, 2*erosion_size+1 ),Point( erosion_size, erosion_size ) );
	erode( dst, erosion_dst, element );

	erosion_size=6;
	element = getStructuringElement( erosion_type, Size( 2*erosion_size + 1, 2*erosion_size+1 ),Point( erosion_size, erosion_size ) );
	dilate( erosion_dst, erosion_dst, element );
	
	vector<Vec4i> hierarchy;
	vector<vector<Point> > contours;
	
	Canny( erosion_dst, erosion_dst, 80, 80*2, 3 );
	findContours(erosion_dst,contours,hierarchy,CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE);
	if(contours.size()==0)
	{
		imshow(window_name,img);
		return;
	}
	int largest_area=0;
	int largest_contour_index=0;
	Rect bounding_rect;
	
	vector<Point2f>center( contours.size() );
	vector<float>radius( contours.size() );
	vector<vector<Point> > contours_poly( contours.size());
	vector<Rect> boundRect( contours.size());

	int j;
	for( int i = 0; i< contours.size(); i++ )
	{
		double a=contourArea( contours[i],false);
		if(a>largest_area){
			approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
			boundRect[i] = boundingRect( Mat(contours_poly[i]) );
			minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
			//largest_area=a;
			//largest_contour_index=i;                
			bounding_rect=boundingRect(contours[i]);  
			j=i;
		}
	}
	center_points.push_back(center[j]);
	Mat drawing = Mat::zeros( erosion_dst.size(), CV_8UC3 );
	Scalar color( 0,255,255);
	drawContours( drawing, contours,largest_contour_index, color, 2, 8, hierarchy );
	Mat cropped;
	cropped = img(bounding_rect).clone();
	imwrite("cropped_pic.jpg",cropped);
	
	logo_detector(bounding_rect);

	if(found_logo==false)
		return;

	img=imread("picture005.jpg");
	rectangle(img, bounding_rect,  Scalar(0,255,0),2, 8,0);
	//circle( img, center[j], (int)radius[j], color, 2, 8, 0 );
	rectangle(img,Point((int)center[j].x-((int)(radius[j])),(int)center[j].y-((int)(2*radius[j]))),Point((int)center[j].x+((int)(radius[j])),(int)center[j].y+((int)(radius[j]*4))),color,2,8,0);
	//rectangle(img,Point((int)center[j].x-(int)((radius[j])/1.5),(int)center[j].y-(int)(radius[j]/1.5)),Point((int)center[j].x+(int)((radius[j]/2)),(int)center[j].y),color,2,8,0);
	imshow(window_name,img);
}