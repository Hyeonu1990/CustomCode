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
			switch (i)
			{
			case 0: //51,39
				DrawingBarcodes(src, data[i], Point(51, 39), true);
				break;
			case 1: // 105, 39
				DrawingBarcodes(src, data[i], Point(105, 39), true);
				break;
			case 2: //165,57
				DrawingBarcodes(src, data[i], Point(165, 57), false);
				break;
			case 3: // 165, 111
				DrawingBarcodes(src, data[i], Point(165, 111), false);
				break;
			case 4: //111 165
				DrawingBarcodes(src, data[i], Point(111, 165), true);
				break;
			case 5: //57 165
				DrawingBarcodes(src, data[i], Point(57, 165), true);
				break;
			case 6: // 39, 105
				DrawingBarcodes(src, data[i], Point(39, 105), false);
				break;
			case 7: // 39, 51
				DrawingBarcodes(src, data[i], Point(39, 51), false);
				break;
			default:
				break;
			}
		}
		imshow("insertdata", *src);
	}
private:

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