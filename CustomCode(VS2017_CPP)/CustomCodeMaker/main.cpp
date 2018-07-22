#include "codeMaker.h"
void main()
{
	Mat input = imread("./img/dodo.png");
	Mat* base = new Mat(imread("./img/210x210base.png")); 
	Mat output;

	//cvtColor(base, base, input.type());
	resize(input, output, Size(210, 210), 0, 0, CV_INTER_LINEAR);	

	unsigned char data[8] = { '0','1','2','3','4','5','6','7' };
	codeMaker codemaker;
	codemaker.InsertData(base, data);
	imshow("base", *base);

	for(int x = 0; x < output.rows; x++)
		for (int y = 0; y < output.cols; y++)
		{
			if (base->at<Vec3b>(y, x)[0] == 0)
			{
				for (int i = 0; i < 3; i++)
					output.at<Vec3b>(y, x)[i] = 0;
			}
			else if (base->at<Vec3b>(y, x)[0] == 255)
			{
				for (int i = 0; i < 3; i++)
					output.at<Vec3b>(y, x)[i] = 255;
			}
		}
	imshow("output", output);
	imwrite("./img/output.png", output);
	waitKey(0);
}


