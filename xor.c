#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// NOTES
// this train is for the xor. You cant use it for nand / or

float train[][3] = {
	{0,0, 0},
	{1,0, 1},
	{0,1, 1},
	{1,1, 0},
};

#define train_count (int)sizeof(train) / (int)sizeof(train[0])

#define h 1e-3
#define N 10000
#define rate 1e-3


struct OR {
	float w1;
	float w2;
	float b;
};

struct NAND {
	float w1;
	float w2;
	float b;
};

struct AND {
	float w_or;
	float w_nand;
	float b;
};

float sigmoid(float x) {
	return 1.f/(1.f + expf(-x));
}

float rand_float() {
	return (float) rand() / (float) RAND_MAX; 
}

float loss(float w1, float w2, float b) {
	float e = 0.f;

	for (int i=0; i < train_count; i++) {
		float x1 = train[i][0];
		float x2 = train[i][1];

		float exp_out = train[i][2];
		float out = sigmoid(w1 * x1 + w2 * x2 + b);

		e = (out - exp_out);
		e *= e;
	}
	e /= train_count;

	return e;
}

float *find_param(float w1, float w2, float b) {
	static float param[3];

	for (int i=0; i < N; i++) {
		float dw1 = (loss(w1 + h, w2, b) - loss(w1, w2, b)) / h;
		float dw2 = (loss(w1, w2 + h, b) - loss(w1, w2, b)) / h;
		float db = (loss(w1, w2, b + h) - loss(w1, w2, b)) / h;

		w1 -= rate * dw1;
		w2 -= rate * dw2;
		b -= rate * db;
	}

	param[0] = w1;
	param[1] = w2;
	param[2] = b;
	return param;
}

float hidden_layer(float y_or[], float y_nand[]) {
	for (int i=0; i < train_count; i++) {
		float x1 = y_or[i];
		float x2 = y_nand[i];
	
		float exp_out = train[i][2];
		float out = sigmoid(w1 * x1 + w2 * x2 + b);

		e = (out - exp_out);
		e *= e;
	}
	e /= train_count;
}

int main() {
	/* float y_and = and.w_or * y_or + and.w_nand * y_nand + and.b; */

	float *param_or = find_param(0.1,0.2,0.2);
	float *param_nand = find_param(0.2,0.2,0.3);

	struct OR or;
	or.w1 = param_or[0];
	or.w2 = param_or[1];
	or.b = param_or[0];

	struct NAND nand;
	nand.w1 = param_nand[0];
	nand.w2 = param_nand[1];
	nand.b = param_nand[2];

	float y_or[4];
	float y_nand[4];
	for (int i=0; i < train_count; i++) {
		y_or[i] = or.w1 * train[i][0] + or.w2 * train[i][1] + or.b;
		y_nand[i] = nand.w1 * train[i][0] + nand.w2 * train[i][1] + nand.b;
	}

	hidden_layer(y_or, y_nand);

	/* float *param_and = find_param(y_or, y_nand, ) */

	/* printf("OR ---  w1: %f  w2: %f  b: %f\n", param_or[0], param_or[1], param_or[2]); */
	/* printf("NAND -  w1: %f  w2: %f  b: %f\n", param_nand[0], param_nand[1], param_nand[2]); */
	/* printf("%f", y_nand[3]); */

	return 0;
}
