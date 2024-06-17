#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

float loss(float w1, float w2, float b) {
	float e = 0.0f;
	for (int i=0; i < train_count; i++) {
		float x1 = train[i][0];
		float x2 = train[i][1];
		float y = w1 * x1 + w2 * x2 + b;

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
	float rate = 1e-3;
	
	float h = 1e-4;

	for (int i=0; i < 30000; i++) {
		float dw1 = (loss(w1 + h, w2, b) - loss(w1, w2, b)) / h;
		float dw2 = (loss(w1, w2 + h, b) - loss(w1, w2, b)) / h;
		float db = (loss(w1, w2, b + h) - loss(w1, w2, b)) / h;
		w1 -= rate * dw1;
		w2 -= rate * dw2;
		b -= rate * db;

		printf("loss = %f, w1 = %f, w2 = %f\n", loss(w1, w2, b), w1, w2);
	}
	
	float out0 = w1 * train[0][0] + w2 * train[0][1] + b;
    float out1 = w1 * train[1][0] + w2 * train[1][1] + b;
    float out2 = w1 * train[2][0] + w2 * train[2][1] + b;
    float out3 = w1 * train[3][0] + w2 * train[3][1] + b;

	printf("------------------------------------------------\n");
	printf("0 and 0: exp 0 found %f\n", out0);
	printf("1 and 0: exp 0 found %f\n", out1);
	printf("0 and 1: exp 0 found %f\n", out2);
	printf("1 and 1: exp 1 found %f\n", out3);

	printf("------------------------------------------------\n");
	printf("1 for true and 0 for false: ");
	int in0;
	int in1;
	scanf("%d", &in0);
	scanf("%d", &in1);
	float out = w1 * in0 + w2 * in1 +b;

	if (out >= 0.5) out = 1;
	else out = 0;
	printf("%d and %d = %.0f", in0, in1, out);

	return 0;
}
