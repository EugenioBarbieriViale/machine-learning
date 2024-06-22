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
#define rate 10

typedef struct {
	float or_w1;
	float or_w2;
	float or_b;

	float nand_w1;
	float nand_w2;
	float nand_b;

	float and_w1;
	float and_w2;
	float and_b;
} Xor;

float sigmoid(float x) {
	return 1.f/(1.f + expf(-x));
}

float rand_float(void) {
	return (float) rand() / (float) RAND_MAX;
}

float forward(Xor m, float x1, float x2) {
	float y_or = sigmoid(m.or_w1 * x1 + m.or_w2 * x2 + m.or_b);
	float y_nand = sigmoid(m.nand_w1 * x1 + m.nand_w2 * x2 + m.nand_b);
	return sigmoid(m.and_w1 * y_or + m.and_w2 * y_nand + m.and_b);
}

float loss(Xor m) {
	float e = 0.f;
	for (int i=0; i<train_count; i++) {
		float x1 = train[i][0];
		float x2 = train[i][1];

		float out = forward(m, x1, x2);
		e += (train[i][2] - out) * (train[i][2] - out);
	}
	e /= train_count;
	
	return e;
}

Xor rand_xor(void) {
	Xor m;
	m.or_w1 = rand_float();
	m.or_w2 = rand_float();
	m.or_b = rand_float();

	m.nand_w1 = rand_float();
	m.nand_w2 = rand_float();
	m.nand_b = rand_float();

	m.and_w1 = rand_float();
	m.and_w2 = rand_float();
	m.and_b = rand_float();
	return m;
}

Xor apply_diff(Xor m, Xor g) {
	m.or_w1 -= rate * g.or_w1;
	m.or_w2 -= rate * g.or_w2;
	m.or_b -= rate * g.or_b;

	m.nand_w1 -= rate * g.nand_w1;
	m.nand_w2 -= rate * g.nand_w2;
	m.nand_b -= rate * g.nand_b;

	m.and_w1 -= rate * g.and_w1;
	m.and_w2 -= rate * g.and_w2;
	m.and_b -= rate * g.and_b;

	return m;
}

// Find the gradient
Xor finite_diff(Xor m) {
	Xor g;
	float prev_loss = loss(m);
	float saved;

	saved = m.or_w1;
	m.or_w1 += h;
	g.or_w1 = (loss(m) - prev_loss) / h;
	m.or_w1 = saved;

	saved = m.or_w2;
	m.or_w2 += h;
	g.or_w2 = (loss(m) - prev_loss) / h;
	m.or_w2 = saved;

	saved = m.or_b;
	m.or_b += h;
	g.or_b = (loss(m) - prev_loss) / h;
	m.or_b = saved;

	saved = m.nand_w1;
	m.nand_w1 += h;
	g.nand_w1 = (loss(m) - prev_loss) / h;
	m.nand_w1 = saved;

	saved = m.nand_w2;
	m.nand_w2 += h;
	g.nand_w2 = (loss(m) - prev_loss) / h;
	m.nand_w2 = saved;

	saved = m.nand_b;
	m.nand_b += h;
	g.nand_b = (loss(m) - prev_loss) / h;
	m.nand_b = saved;

	saved = m.and_w1;
	m.and_w1 += h;
	g.and_w1 = (loss(m) - prev_loss) / h;
	m.and_w1 = saved;

	saved = m.and_w2;
	m.and_w2 += h;
	g.and_w2 = (loss(m) - prev_loss) / h;
	m.and_w2 = saved;

	saved = m.and_b;
	m.and_b += h;
	g.and_b = (loss(m) - prev_loss) / h;
	m.and_b = saved;

	return g;
}

int main() {
	Xor m = rand_xor();

	for (int i=0; i<N; i++) {
		Xor g = finite_diff(m);
		m = apply_diff(m, g);
		/* printf("%f\n", loss(m)); */
	}

	for (int i=0; i<train_count; i++) {
		float in1 = train[i][0];
		float in2 = train[i][1];

		float out = forward(m, in1, in2);
		printf("%0.f | %0.f = %0.f -> %f\n", in1, in2, train[i][2], out);
	}

	return 0;
}
