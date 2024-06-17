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

float loss(float w1, float w2) {
	float e = 0.0f;
	for (int i=0; i < train_count; i++) {
		float x1 = train[i][0];
		float x2 = train[i][1];
		float y = w1 * x1 + w2 * x2;

		e += (train[i][2] - y) * (train[i][2] - y);
	}
	e /= train_count;

	return e;
}

int main() {
	/* srand(time(0)); */
	srand(438);
	float w1 = rand_float()*3;
	float w2 = rand_float()*5;
	float rate = 1e-3;
	
	float h = 1e-3;

	for (int i=0; i < 30000; i++) {
		float dw1 = (loss(w1 + h, w2) - loss(w1, w2)) / h;
		float dw2 = (loss(w1, w2 + h) - loss(w1, w2)) / h;
		w1 -= rate * dw1;
		w2 -= rate * dw2;

		printf("loss = %f, w1 = %f, w2 = %f\n", loss(w1, w2), w1, w2);
	}

	return 0;
}
