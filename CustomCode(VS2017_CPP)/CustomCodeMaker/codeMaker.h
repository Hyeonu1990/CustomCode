#pragma once
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;

class codeMaker
{
public:

	void InsertData(Mat* src, unsigned char* data)
	{
		for (int i = 0; i < 8; i++)
		{
			printf("data[%d] : %d\n", i, data[i]);
			DrawingBarcodes(src, data[i], pos[i], (i == 0  || i == 1 || i == 4 || i == 5)?true:false);			
		}
		imshow("insertdata", *src);
	}
private:
	Point pos[8] = { Point(51, 39), Point(105, 39), Point(165, 57), Point(165, 111), Point(111, 165), Point(57, 165), Point(39, 105), Point(39, 51) };
	unsigned char bit = 128;
	unsigned char tmp = 0;	

	void DrawingBarcodes(Mat* src, unsigned char data, Point pos, bool horizontal)
	{
		for (int n = 0; n < 8; n++)
		{
			tmp = data;
			tmp &= bit;
			printf("tmp : %d, bit : %d\n", tmp,bit);
			if (tmp == 0)
			{
				for (int x = 0; x < 6; x++)
					for (int y = 0; y < 6; y++)
					{
						src->at<Vec3b>(pos.y + y + ((!horizontal)?(n * 6):0), pos.x + x + ((horizontal)?(n * 6):0) )[0] = 0;
					}
			}
			bit = bit >> 1;
		}
		bit = 128;
	}
};