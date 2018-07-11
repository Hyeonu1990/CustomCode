#pragma once
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>

using namespace cv;
using namespace std;

class CustomCode
{
public :
	bool pcmode = true;
	bool smallMode = true;
	int i = 0;
	void recognition(Mat* src, vector<Point2f>* mker, string* result)
	{
		vector<Point2f> pmarker;

		int angle = Find_Code(src, &pmarker);
		
		if (pmarker.size() > 0 && angle != -1)
		{
			if (pmarker.size() == 4)
			{
				vector<Point2f> entireMarker = pmarker;
				*mker = entireMarker;
				//�밢��
				Point2f v1 = VectorNomalize(entireMarker[2] - entireMarker[0]);
				Point2f v2 = VectorNomalize(entireMarker[3] - entireMarker[1]);
				float nj = v1.x * v2.x + v1.y * v2.y;
				//�� ������
				Point2f vv1 = VectorNomalize(entireMarker[3] - entireMarker[0]);
				Point2f vv2 = VectorNomalize(entireMarker[3] - entireMarker[2]);
				float NEnj = v1.x * v2.x + v1.y * v2.y;
				//�Ʒ� ����
				Point2f vvv1 = VectorNomalize(entireMarker[1] - entireMarker[0]);
				Point2f vvv2 = VectorNomalize(entireMarker[1] - entireMarker[2]);
				float SWnj = v1.x * v2.x + v1.y * v2.y;

				if (abs(nj) < 0.18f && // 0.04
					(Angle(NEnj) > 85 && Angle(NEnj) < 96) &&
					(Angle(SWnj) > 80 && Angle(SWnj) < 95))
				{


					Mat marker_image;
					int marker_image_side_length = smallMode ? 210 : 354;//smallMode?288:576; 

					vector<Point2f> square_points;
					square_points.push_back(Point2f(0, 0));
					square_points.push_back(Point2f(0, marker_image_side_length - 1)); //Point(marker_image_side_length - 1, 0)
					square_points.push_back(Point2f(marker_image_side_length - 1, marker_image_side_length - 1));
					square_points.push_back(Point2f(marker_image_side_length - 1, 0)); //Point(0, marker_image_side_length - 1)

					//��Ŀ�� �簢�����·� �ٲ� perspective transformation matrix�� ���Ѵ�.
					vector<Point2f> dst;
					dst.push_back(Point2f((float)entireMarker[0].x, (float)entireMarker[0].y));
					dst.push_back(Point2f((float)entireMarker[1].x, (float)entireMarker[1].y)); //Point2f((float)entireMarker[3].x, (float)entireMarker[3].y)
					dst.push_back(Point2f((float)entireMarker[2].x, (float)entireMarker[2].y));
					dst.push_back(Point2f((float)entireMarker[3].x, (float)entireMarker[3].y)); //Point2f((float)entireMarker[1].x, (float)entireMarker[1].y)
					
					Mat PerspectiveTransformMatrix = getPerspectiveTransform(entireMarker, dst);
					
					//perspective transformation�� �����Ѵ�. 
					Mat input_gray_image;
					cvtColor(*src, input_gray_image, 6);

					//if (pcmode) Imgcodecs.imwrite("D:\\imgs\\input_gray_image" + i.ToString() + ".png", input_gray_image);

					//Imgproc.warpPerspective(input_gray_image, marker_image, PerspectiveTransformMatrix, new Size(marker_image_side_length, marker_image_side_length));
					warpPerspective(*src, marker_image, PerspectiveTransformMatrix, Size(marker_image_side_length, marker_image_side_length));

					//�Կ��� �̹��� �ҷ����� �κ�(������)
					//cvtColor(marker_image, marker_image, Imgproc.COLOR_BGR2RGB);
					//if (pcmode) imwrite("D:\\imgs\\total_marker_image" + i.ToString() + ".png", marker_image);

					Mat valueimg = marker_image(Rect(33, 33, marker_image_side_length - 33 * 2, marker_image_side_length - 33 * 2));
					//Imgproc.threshold(valueimg, valueimg, 0, 255, 0 | 8); // THRESH_BINARY = 0 | THRESH_OTSU = 8
					adaptiveThreshold(valueimg, valueimg, 255, 1, 0, 35, 2); // ADAPTIVE_THRESH_GAUSSIAN_C = 1, THRESH_BINARY = 0
					if (pcmode) imwrite("D:\\imgs\\valueimg" + to_string(i) + ".png", valueimg);


					//���� 10x10�� �ִ� ������ ��Ʈ�� �����ϱ� ���� ����
					int cellSize = smallMode ? 6 : 12;//6;
					int bitmatrix_size = (marker_image_side_length - 33 * 2) / cellSize;//smallMode ? (144 / cellSize) : (marker_image_side_length * 3 / 4 / cellSize); //144 / cellSize;
					Mat bitMatrix = Mat::zeros(bitmatrix_size, bitmatrix_size, CV_8UC1); // 144 / 3 = 48

					for (int y = 0; y < bitmatrix_size; y++)
					{
						for (int x = 0; x < bitmatrix_size; x++)
						{
							int cellX = x * cellSize;
							int cellY = y * cellSize;
							Mat cell = valueimg(Rect(cellX, cellY, cellSize, cellSize));

							int total_cell_count = countNonZero(cell);

							if (total_cell_count >(cellSize * cellSize) / 2)
							{
								bitMatrix.at<uchar>(y, x) = 255;
							}
						}
					}
					if (pcmode) imwrite("D:\\imgs\\valueimg_bitMatrix" + to_string(i) + ".png", bitMatrix);
					*result = CatchCharArray(bitMatrix);
					i++;
				}
			}
		}
	}

private:
	int Find_Code(Mat* src, vector<Point2f>* mk)
	{
		Mat input_image = src->clone();
		if (pcmode) imwrite("D:\\imgs\\input_image" + to_string(i) + ".png", input_image);

		Mat input_gray_image;

		cvtColor(input_image, input_gray_image, 6);

		//Adaptive Thresholding�� �����Ͽ� ����ȭ �Ѵ�. 
		Mat binary_image;
		adaptiveThreshold(input_gray_image, binary_image, 255, 1, 1, 37, 2); //ADAPTIVE_THRESH_GAUSSIAN_C = 1, THRESH_BINARY_INV = 1, 91, 7
																					 //Imgproc.threshold(input_gray_image, binary_image, 0, 255, 1 | 8); //THRESH_BINARY_INV | THRESH_OTSU
		if (pcmode) imwrite("D:\\imgs\\binary_image" + to_string(i) + ".png", binary_image);
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////�ڵ�簢������

		//contours�� ã�´�.
		Mat contour_image = binary_image.clone();
		vector<vector<Point>> contours;
		Mat hierarchy;
		findContours(contour_image, contours, hierarchy, 1, 2); //RETR_LIST, CHAIN_APPROX_SIMPLE

																		//contour�� �ٻ�ȭ�Ѵ�.
		vector<vector<Point2f>> marker; //�ؿ� for������ contours���� ���ǿ� �´� �༮�� ����� ����
		vector<vector<Point2f>> L5marker; //maker�� �����
		vector<vector<Point2f>> L1marker; //maker�� �����
		vector<Point2f> approx;

		vector<Point2f> L5pos; //����绩�� ��ġ��(5���� �簢�� �𼭸�)
		Point L1pos; // ����� ��ġ(1���� �簢�� �𼭸�)

		for (int i = 0; i < contours.size(); i++)
		{
			approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.05, true);
			if (
				fabs(contourArea(Mat(approx))) > input_image.rows / 30 * input_image.cols / 30 && //������ ����ũ�� �̻��̾�� �Ѵ�. //input_image.rows() / 10 * input_image.cols() / 10 
				fabs(contourArea(Mat(approx))) < input_image.rows / 7 * input_image.cols / 7 //������ ����ũ�� ���Ͽ��� �Ѵ�. //input_image.rows() / 4 * input_image.cols() / 4
				)
			{
				if (approx.size() == 4 && //�簢���� 4���� vertex�� ������.
					isContourConvex(Mat(approx)) //convex���� �˻��Ѵ�.
					)
				{
					vector<Point2f> points;
					for (int j = 0; j<approx.size(); j++)
						points.push_back(cv::Point2f(approx[j].x, approx[j].y));
					marker.push_back(points);
				}
				else if (approx.size() == 6 && //�������� 6���� vertex�� ������.
					!isContourConvex(Mat(approx)) //convex�� �ƴ��� �˻��Ѵ�.
					)
				{
					//���� ������ ���� ������
					Point2f vec0 = approx[0];
					Point2f vec1 = approx[2];
					Point2f vec2 = approx[4];
					Point2f vec3 = approx[5];
					float vnj = VectorNomalize(vec2 - vec0).x * VectorNomalize(vec3 - vec1).x + VectorNomalize(vec2 - vec0).y * VectorNomalize(vec3 - vec1).y;
					if (abs(vnj) < 0.46)// ������� ������ ������ //0.17
					{
						vector<Point2f> points;
						for (int j = 0; j<approx.size(); j++)
							points.push_back(cv::Point2f(approx[j].x, approx[j].y));
						L5marker.push_back(points);
						//L5pos.Add(points.toArray()[5]);
						//Debug.Log("L5pos : " + points.toArray()[5] + "\n vnj : " + Mathf.Abs(vnj) + "\n angle : " + angle);
						//Debug.Log(points.toArray()[5] +"\n"+(input_image.rows() / 30 * input_image.cols() / 30) + ", " + Imgproc.contourArea(approx));
					}
					else if (abs(vnj) > 0.9) // ����� ������
					{
						vector<Point2f> points;
						for (int j = 0; j<approx.size(); j++)
							points.push_back(cv::Point2f(approx[j].x, approx[j].y));
						L1marker.push_back(points);
						//L1pos = points.toArray()[1];
						//Debug.Log("L1pos : " + points.toArray()[5] + "\n vnj : " + Mathf.Abs(vnj) + "\n angle : " + angle);
					}
				}
			}
		}

		//Imgcodecs.imwrite("D:\\imgs\\input_image.png", input_image);
		////////�簢��P1��Ȯ��/////////////////////////////////////////////////////////////////////////////////////////////////////////////�ڵ�簢������

		vector<vector<Point2f>>detectedMarkers;
		vector<Mat> detectedMarkersImage;

		int marker_image_side_length = 100; //P��Ŀ 10x10
											//�̹����� ���ڷ� ������ �� ���ϳ��� �ȼ��ʺ� 10���� �Ѵٸ�
											//P��Ŀ �̹����� �Ѻ� ���̴� 100
		vector<Point2f> square_points;
		square_points.push_back(Point2f(0, 0));
		square_points.push_back(Point2f(0, marker_image_side_length - 1));
		square_points.push_back(Point2f(marker_image_side_length - 1, marker_image_side_length - 1));
		square_points.push_back(Point2f(marker_image_side_length - 1, 0));

		Mat marker_image;
		for (int i = 0; i < marker.size(); i++)
		{
			vector<Point2f> m = marker[i];
			//��Ŀ�� �簢�����·� �ٲ� perspective transformation matrix�� ���Ѵ�.
			Mat PerspectiveTransformMatrix = getPerspectiveTransform(m, square_points);

			//perspective transformation�� �����Ѵ�. 
			warpPerspective(input_gray_image, marker_image, PerspectiveTransformMatrix, Size(marker_image_side_length, marker_image_side_length));
			//imshow("input_gray_image_pserspective_transform" + to_string(i), marker_image);

			//����ȭ�� �����Ѵ�. 
			threshold(marker_image, marker_image, 0, 255, 0 | 8); // THRESH_BINARY = 0 | THRESH_OTSU = 8
																		  //Imgproc.adaptiveThreshold(marker_image, marker_image, 255, 1, 0, 11, 2); // ADAPTIVE_THRESH_GAUSSIAN_C = 1, THRESH_BINARY = 0
																		  //if (pcmode) Imgcodecs.imwrite("D:\\imgs\\marker_image" + i.ToString() + ".png", marker_image);

																		  //�׵θ��˻�κ�
																		  //������ �µθ��� ������ ũ��� 10
																		  //��Ŀ �̹��� �׵θ��� �˻��Ͽ� ���� ���������� Ȯ���Ѵ�. 
			int cellSize = marker_image.rows / 10;
			int white_cell_count = 0;
			for (int y = 0; y < 10; y++)
			{
				int inc = 10; // ù��° ���� ������ ���� �˻��ϱ� ���� ��

				if (y == 0 || y == 9) inc = 1; //ù��° �ٰ� ���������� ��� ���� �˻��Ѵ�. 

				for (int x = 0; x < 10; x += inc)
				{
					int cellX = x * cellSize;
					int cellY = y * cellSize;
					Mat cell = marker_image(Rect(cellX, cellY, cellSize, cellSize));

					int total_cell_count = countNonZero(cell);

					if (total_cell_count >(cellSize * cellSize) / 2)
						white_cell_count++; //�µθ��� ��������� �ִٸ�, ������ �ȼ��� �����̻� ����̸� ����������� ����
				}
			}
			//������ �µθ��� �ѷ��׿� �ִ� �͸� �����Ѵ�.
			if (white_cell_count == 0)
			{
				detectedMarkers.push_back(m);
				Mat img = marker_image.clone();
				detectedMarkersImage.push_back(img);
			}
		}
		//marker = detectedMarkers;

		////////L1��Ȯ��/////////////////////////////////////////////////////////////////////////////////////////////////////////////�ڵ�簢������

		int Lmarker_image_side_length = 100; //L��Ŀ 3x10
		vector<Point2f> Lsquare_points;
		Lsquare_points.push_back(Point2f(0, 0));
		Lsquare_points.push_back(Point2f(0, Lmarker_image_side_length * 3 / 10 - 1));
		Lsquare_points.push_back(Point2f(Lmarker_image_side_length - 1, Lmarker_image_side_length * 3 / 10 - 1));
		Lsquare_points.push_back(Point2f(Lmarker_image_side_length - 1, 0));
		//Mat marker_image = new Mat();
		for (int i = 0; i < L5marker.size(); i++)
		{
			vector<Point2f> m = L5marker[i];
			//��Ŀ�� �簢�����·� �ٲ� perspective transformation matrix�� ���Ѵ�.
			vector<Point2f> persrc;
			persrc.push_back(Point2f((float)m[0].x, (float)m[0].y));
			persrc.push_back(Point2f((float)m[1].x, (float)m[1].y));
			persrc.push_back(Point2f((float)m[2].x + (float)m[4].x - (float)m[3].x, (float)m[2].y + (float)m[4].y - (float)m[3].y));
			persrc.push_back(Point2f((float)m[5].x, (float)m[5].y));
			
			Mat PerspectiveTransformMatrix = getPerspectiveTransform(persrc, Lsquare_points);

			//perspective transformation�� �����Ѵ�. 
			warpPerspective(input_gray_image, marker_image, PerspectiveTransformMatrix, Size(Lmarker_image_side_length, Lmarker_image_side_length * 3 / 10));
			//imshow("input_gray_image_pserspective_transform" + to_string(i), marker_image);

			//����ȭ�� �����Ѵ�. 
			threshold(marker_image, marker_image, 0, 255, 0 | 8); // THRESH_BINARY = 0 | THRESH_OTSU = 8
																		  //Imgproc.adaptiveThreshold(marker_image, marker_image, 255, 1, 0, 11, 2); // ADAPTIVE_THRESH_GAUSSIAN_C = 1, THRESH_BINARY = 0

			int cellSize = marker_image.cols / 10;
			int white_cell_count = 0;
			char data = 0;
			char tmp = 1;

			for (int x = 1; x < 9; x += 1)
			{
				int cellX = x * cellSize;
				int cellY = 1 * cellSize;
				Mat cell = marker_image(Rect(cellX, cellY, cellSize, cellSize));

				int total_cell_count = countNonZero(cell);

				if (total_cell_count >(cellSize * cellSize) / 2)
				{
					white_cell_count++; //�µθ��� ��������� �ִٸ�, ������ �ȼ��� �����̻� ����̸� ����������� ����
					data |= tmp;
				}
				tmp <<= 1;
			}
			//if (pcmode) Imgcodecs.imwrite("D:\\imgs\\L5marker_image_" + i + "_" + white_cell_count + ".png", marker_image);
			//Debug.Log("L5pos : "+m.toArray()[5] + "\ndata : " + (int)data);
			if (data == 171 || data == 175)//(white_cell_count == 5 || white_cell_count == 6)
			{
				L5pos.push_back(m[5]);
			}
		}


		for (int i = 0; i < L1marker.size(); i++)
		{
			vector<Point2f> m = L1marker[i];
			//��Ŀ�� �簢�����·� �ٲ� perspective transformation matrix�� ���Ѵ�.
			vector<Point2f> persrc;
			persrc.push_back(Point2f((float)m[4].x - (float)m[5].x + (float)m[0].x, (float)m[4].y - (float)m[5].y + (float)m[0].y));
			persrc.push_back(Point2f((float)m[1].x, (float)m[1].y));
			persrc.push_back(Point2f((float)m[2].x, (float)m[2].y));
			persrc.push_back(Point2f((float)m[3].x, (float)m[3].y));
			Mat PerspectiveTransformMatrix = getPerspectiveTransform(persrc, Lsquare_points);

			//perspective transformation�� �����Ѵ�. 
			warpPerspective(input_gray_image, marker_image, PerspectiveTransformMatrix, Size(Lmarker_image_side_length, Lmarker_image_side_length * 3 / 10));
			//imshow("input_gray_image_pserspective_transform" + to_string(i), marker_image);

			//����ȭ�� �����Ѵ�. 
			threshold(marker_image, marker_image, 0, 255, 0 | 8); // THRESH_BINARY = 0 | THRESH_OTSU = 8
																		  //Imgproc.adaptiveThreshold(marker_image, marker_image, 255, 1, 0, 11, 2); // ADAPTIVE_THRESH_GAUSSIAN_C = 1, THRESH_BINARY = 0

			int cellSize = marker_image.cols / 10;
			int white_cell_count = 0;
			char data = 0;
			char tmp = 1;

			for (int x = 1; x < 9; x += 1)
			{
				int cellX = x * cellSize;
				int cellY = 1 * cellSize;
				Mat cell = marker_image(Rect(cellX, cellY, cellSize, cellSize));

				int total_cell_count = countNonZero(cell);

				if (total_cell_count >(cellSize * cellSize) / 2)
				{
					white_cell_count++; //�µθ��� ��������� �ִٸ�, ������ �ȼ��� �����̻� ����̸� ����������� ����
					data |= tmp;
				}
				tmp <<= 1;
			}
			//if (pcmode) Imgcodecs.imwrite("D:\\imgs\\L1marker_image_" + i + "_" + white_cell_count + ".png", marker_image);
			//Debug.Log("L1pos : " + m.toArray()[5] + "\ndata : " + (int)data);
			if (data == 223)//(white_cell_count == 7)
				L1pos = m[1];
		}


		/////��Ŀ�� ��Ʈ��Ʈ����ȭ �Ͽ� �����Ǵ�////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//������ �׵θ� ������ ���� Ȯ��

		list<Mat> bitMatrixs;

		for (int i = 0; i < detectedMarkers.size(); i++)
		{
			//cout << "Black Line Checking Start" << endl;
			Mat marker_img = detectedMarkersImage[i];

			//���� 10x10�� �ִ� ������ ��Ʈ�� �����ϱ� ���� ����
			Mat bitMatrix = Mat::zeros(10, 10, CV_8UC1);

			int cellSize = marker_img.rows / 10;
			//cout << "Marker" + to_string(i) << endl;
			int white_cell_count = 1;
			int black_cell_count = 0;
			int angle = 0;

			for (int y = 1; y < 9; y++)
			{
				int inc = 7; // ù��° ���� ������ ���� �˻��ϱ� ���� ��

				if (y == 1 || y == 8) inc = 1; //ù��°(0)�� ����(1) �ٰ� ������(9)�� ��(8)���� ��� ���� �˻��Ѵ�. 

				for (int x = 1; x < 9; x += inc)
				{
					int cellX = x * cellSize;
					int cellY = y * cellSize;
					Mat cell = marker_img(Rect(cellX, cellY, cellSize, cellSize));

					int total_cell_count = countNonZero(cell);

					if (total_cell_count < (cellSize * cellSize) / 2)
					{
						black_cell_count++;
					}
				}
			}

			//ȸ��Ȯ���� ���� ��Ʈȭ
			if (black_cell_count == 0)
			{
				for (int y = 0; y < 10; y++)
				{
					for (int x = 0; x < 10; x++)
					{
						int cellX = x * cellSize;
						int cellY = y * cellSize;
						Mat cell = marker_img(Rect(cellX, cellY, cellSize, cellSize));

						int total_cell_count = countNonZero(cell);

						if (total_cell_count >(cellSize * cellSize) / 2)
						{
							bitMatrix.at<uchar>(y, x) = 255;
						}
					}
				}
				if (pcmode) imwrite("D:\\imgs\\bitMatrix" + to_string(i) + ".png", bitMatrix);
				//�����Ǵ�
				//if (bitMatrix.get(7, 7)[0] > 0 && bitMatrix.get(7, 6)[0] == 0)
				if (bitMatrix.at<uchar>(2, 7) > 0 && bitMatrix.at<uchar>(2, 6) == 0)
				{
					//-90
					angle = -90;
					Mat rotationimg;
					rotate(bitMatrix, rotationimg, 0); // +90
					bitMatrix = rotationimg.clone();
				}
				//else if (bitMatrix.get(7, 2)[0] > 0 && bitMatrix.get(6, 2)[0] == 0)
				else if (bitMatrix.at<uchar>(7, 7) > 0 && bitMatrix.at<uchar>(6, 7) == 0)
				{
					//0
					angle = 0;
				}
				//else if (bitMatrix.get(2, 7)[0] > 0 && bitMatrix.get(3, 7)[0] == 0)
				else if (bitMatrix.at<uchar>(2, 2) > 0 && bitMatrix.at<uchar>(3, 2) == 0)
				{
					//180
					angle = 180;
					Mat rotationimg;
					rotate(bitMatrix, rotationimg, 1);
					bitMatrix = rotationimg.clone();
				}
				//else if (bitMatrix.get(2, 2)[0] > 0 && bitMatrix.get(2, 2)[0] == 0)
				else if (bitMatrix.at<uchar>(7, 2) > 0 && bitMatrix.at<uchar>(7, 3) == 0)
				{
					//90
					angle = 90;
					Mat rotationimg;
					rotate(bitMatrix, rotationimg, 2); // -90
					bitMatrix = rotationimg.clone();
				}
				else
				{
					continue;
				}
				if (pcmode) imwrite("D:\\imgs\\bitMatrixturned" + to_string(i) + "_" + to_string(angle) + ".png", bitMatrix);
				//P�Ǵ�
				white_cell_count = 0;
				for (int y = 0; y < 6; y++)
				{
					for (int x = 0; x < 6; x++)
					{
						int X = x + 2;
						int Y = y + 2;
						if (
							(X == 2 && Y == 2) ||
							(X == 3 && Y == 2) ||
							(X == 4 && Y == 2) ||
							(X == 5 && Y == 2) ||
							(X == 6 && Y == 2) ||
							(X == 7 && Y == 2) ||
							(X == 2 && Y == 3) ||
							(X == 7 && Y == 3) ||
							(X == 2 && Y == 4) ||
							//(X == 4 && Y == 4) ||
							//(X == 5 && Y == 4) ||
							(X == 7 && Y == 4) ||
							(X == 2 && Y == 5) ||
							(X == 7 && Y == 5) ||
							(X == 2 && Y == 6) ||
							(X == 4 && Y == 6) || // 3
							(X == 5 && Y == 6) || // 4
							(X == 6 && Y == 6) || // 5
							(X == 7 && Y == 6) || // 7
							(X == 2 && Y == 7) || // 5
							(X == 3 && Y == 7) || // 6
							(X == 4 && Y == 7)    // 7
							)
						{
							if (bitMatrix.at<uchar>(Y, X) > 0)
							{
								white_cell_count++;
							}
						}
					}
				}
			}

			if (white_cell_count == 0 && black_cell_count == 0)
			{
				if (L5pos.size() >= 2)
				{
					if (L1pos != Point())
					{
						//0��
						if (L5pos[0].y > L5pos[1].y)
						{
							Point temp = L5pos[0];
							L5pos[0] = L5pos[1];
							L5pos[1] = temp;
						}
						mk->push_back(detectedMarkers[i][0]);
						mk->push_back(L1pos);
						mk->push_back(L5pos[1]);
						mk->push_back(L5pos[0]);
					}
					else if (L5pos.size() == 3)
					{
						//L5pos.Count == 3, -90���϶�
					}
					//Lmarker = Lpos;
				}
				return angle;
			}
		}
		return -1;
	}

	// 8bit ���� char array�� ���� �� string���� ���
	static string CatchCharArray(Mat bitMatrix)
	{
		char* value = new char[9];
		char tmp = 0;
		//1��°
		tmp = 0;
		for (int x = 3; x <= 10; x++)
		{
			//cout << to_string(src.at<uchar>(y, x) > 0 ? 1 : 0) << endl;
			tmp |= (bitMatrix.at<uchar>(1, x) > 0 ? 1 : 0);
			if (x != 10) tmp <<= 1;
		}
		value[0] = tmp;

		//2��°
		tmp = 0;
		for (int x = 12; x <= 19; x++)
		{
			//cout << to_string(src.at<uchar>(y, x) > 0 ? 1 : 0) << endl;
			tmp |= (bitMatrix.at<uchar>(1, x) > 0 ? 1 : 0);
			if (x != 19) tmp <<= 1;
		}
		value[1] = tmp;

		//3��°
		tmp = 0;
		for (int y = 4; y <= 11; y++)
		{
			//cout << to_string(src.at<uchar>(y, x) > 0 ? 1 : 0) << endl;
			tmp |= (bitMatrix.at<uchar>(y, 22) > 0 ? 1 : 0);
			if (y != 11) tmp <<= 1;
		}
		value[2] = tmp;

		//4��°
		tmp = 0;
		for (int y = 13; y <= 20; y++)
		{
			//cout << to_string(src.at<uchar>(y, x) > 0 ? 1 : 0) << endl;
			tmp |= (bitMatrix.at<uchar>(y, 22) > 0 ? 1 : 0);
			if (y != 20) tmp <<= 1;
		}
		value[3] = tmp;

		//5��°
		tmp = 0;
		for (int x = 13; x <= 20; x++)
		{
			//cout << to_string(src.at<uchar>(y, x) > 0 ? 1 : 0) << endl;
			tmp |= (bitMatrix.at<uchar>(22, x) > 0 ? 1 : 0);
			if (x != 20) tmp <<= 1;
		}
		value[4] = tmp;

		//6��°
		tmp = 0;
		for (int x = 4; x <= 11; x++)
		{
			//cout << to_string(src.at<uchar>(y, x) > 0 ? 1 : 0) << endl;
			tmp |= (bitMatrix.at<uchar>(22, x) > 0 ? 1 : 0);
			if (x != 11) tmp <<= 1;
		}
		value[5] = tmp;

		//7��°
		tmp = 0;
		for (int y = 12; y <= 19; y++)
		{
			//cout << to_string(src.at<uchar>(y, x) > 0 ? 1 : 0) << endl;
			tmp |= (bitMatrix.at<uchar>(y, 1) > 0 ? 1 : 0);
			if (y != 19) tmp <<= 1;
		}
		value[6] = (char)tmp;

		//8��°
		tmp = 0;
		for (int y = 3; y <= 10; y++)
		{
			//cout << to_string(src.at<uchar>(y, x) > 0 ? 1 : 0) << endl;
			tmp |= (bitMatrix.at<uchar>(y, 1) > 0 ? 1 : 0);
			if (y != 10) tmp <<= 1;
		}
		value[7] = (char)tmp;

		value[8] = '\0';

		bool BadValue = false;
		for (int i = 0; i < 8; i++)
		{
			if ((value[i] >= 48 && value[i] <= 57) //0~9
				|| (value[i] >= 65 && value[i] <= 90) // A~Z
				|| (value[i] >= 97 && value[i] <= 122)) // a ~ z
			{

			}
			else
			{
				BadValue = true;
				//Debug.Log((int)value[i]);
			}
		}

		if (BadValue)
		{
			//value.Initialize();
			//value[0] = '\0';
			free(value);
			return "";
		}
		else
		{
			return string(value);
		}
	}

	Point2f VectorNomalize(Point2f pt)
	{
		Point2f result;

		result.x = pt.x / sqrt(pow(pt.x, 2) + pow(pt.y, 2));
		result.y = pt.y / sqrt(pow(pt.x, 2) + pow(pt.y, 2));

		return result;
	}

	float Angle(float innerValue)
	{
		int angle = acosf(innerValue) * 180 / CV_PI;

		return angle;
	}
};
