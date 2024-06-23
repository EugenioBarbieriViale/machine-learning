#include <stdio.h>
#include <stdlib.h>
#include <math.h>

float train[][3] = {
	{0,0, 0},
	{1,0, 1},
	{0,1, 1},
	{1,1, 0},
};

#define train_count (int)sizeof(train) / (int)sizeof(train[0])

#define h 1e-3
#define N 10000
#define rate 20

// or_w1 or_w1 or_b -- nand_w1 nand_w2 nand_b -- and_w1 and_w2 and_b
float xor[9]; // Model

// Gradient
float g[9];

float sigmoid(float x) {
	return 1.f/(1.f + expf(-x));
}

float rand_float(void) {
	return (float) rand() / (float) RAND_MAX;
}

float forward(float xor[], float x1, float x2) {
	float y_or = sigmoid(xor[0] * x1 + xor[1] * x2 + xor[2]);
	float y_nand = sigmoid(xor[3] * x1 + xor[4] * x2 + xor[5]);
	return sigmoid(xor[6] * y_or + xor[7] * y_nand + xor[8]);
}

float loss(float xor[]) {
	float e = 0.f;
	for (int i=0; i<train_count; i++) {
		float x1 = train[i][0];
		float x2 = train[i][1];

		float out = forward(xor, x1, x2);
		e += (train[i][2] - out) * (train[i][2] - out);
	}
	e /= train_count;
	
	return e;
}

float *rand_xor() {
	for (int i=0; i<9; i++) {
		xor[i] = rand_float();
	}
	return xor;
}

float *apply_diff(float xor[], float g[]) {
	for (int i=0; i<9; i++) {
		xor[i] -= rate * g[i];
	}
	return xor;
}

// Find the gradient
float *gradient(float xor[]) {
	float prev_loss = loss(xor);
		
	for (int i=0; i<9; i++) {
		xor[i] += h;
		g[i] = (loss(xor) - prev_loss) / h;
		xor[i] -= h;
	}
	return g;
}

int main() {
	float *m;
	m = rand_xor();

	float *g;
	for (int i=0; i<N; i++) {
		g = gradient(m);
		m = apply_diff(m, g);
		printf("%f\n", loss(m));
	}
	printf("-------------------------\n");

	for (int i=0; i<train_count; i++) {
		float in1 = train[i][0];
		float in2 = train[i][1];

		float out = forward(m, in1, in2);
		printf("%0.f | %0.f = %0.f -> %f\n", in1, in2, train[i][2], out);
	}

	return 0;
}
