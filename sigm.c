#include <stdio.h>
#include <stdlib.h>
#include <math.h>

float train[][3] = {
//   A B  A and B
	{0,0, 0},
	{1,0, 0},
	{0,1, 0},
	{1,1, 1},
};

#define train_count (int)sizeof(train) / (int)sizeof(train[0])

float rand_float(void) {
	return (float) rand() / (float) RAND_MAX;
}

float sigmoid(float x) {
	return 1.f/(1.f + expf(-x));
}

float loss(float w1, float w2, float b) {
	float e = 0.0f;
	for (int i=0; i < train_count; i++) {
		float x1 = train[i][0];
		float x2 = train[i][1];
		float y = sigmoid(w1 * x1 + w2 * x2 + b);

		e += (train[i][2] - y) * (train[i][2] - y);
	}
	e /= train_count;

	return e;
}

int main() {
	srand(59);
	float w1 = rand_float()*1.3;
	float w2 = rand_float()*0.9;
	float b = rand_float()*1.2;
	float rate = 20;
	
	float h = 1e-1;

	for (int i=0; i < 30000; i++) {
		float dw1 = (loss(w1 + h, w2, b) - loss(w1, w2, b)) / h;
		float dw2 = (loss(w1, w2 + h, b) - loss(w1, w2, b)) / h;
		float db = (loss(w1, w2, b + h) - loss(w1, w2, b)) / h;
		w1 -= rate * dw1;
		w2 -= rate * dw2;
		b -= rate * db;

		printf("loss = %f, w1 = %f, w2 = %f\n", loss(w1, w2, b), w1, w2);
	}
	
	float out0 = sigmoid(w1 * train[0][0] + w2 * train[0][1] + b);
    float out1 = sigmoid(w1 * train[1][0] + w2 * train[1][1] + b);
    float out2 = sigmoid(w1 * train[2][0] + w2 * train[2][1] + b);
    float out3 = sigmoid(w1 * train[3][0] + w2 * train[3][1] + b);

	printf("------------------------------------------------\n");
	printf("0 and 0: exp 0 found %f\n", out0);
	printf("1 and 0: exp 0 found %f\n", out1);
	printf("0 and 1: exp 0 found %f\n", out2);
	printf("1 and 1: exp 1 found %f\n", out3);

	return 0;
}
