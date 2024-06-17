#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// input - output
float train[][2] = {
	{0,0},
	{1,2},
	{2,4},
	{3,6},
	{4,8},
};

#define train_count (int)sizeof(train) / (int)sizeof(train[0])

float rand_float(void) {
	return (float) rand() / (float) RAND_MAX;
}

float loss(float w, float b) {
	float e = 0.0f;
	for (int i=0; i < train_count; i++) {
		float x = train[i][0];
		float y = w * x + b;

		e += (train[i][1] - y) * (train[i][1] - y);
	}
	e /= train_count;

	return e;
}

int main() {
	/* srand(time(0)); */
	srand(438);
	float w = rand_float()*10;
	float b = rand_float()*5;
	float rate = 1e-3;
	
	float h = 1e-3;

	for (int i=0; i < 200000; i++) {
		float dw = (loss(w + h, b) - loss(w, b)) / h;
		float db = (loss(w, b + h) - loss(w, b)) / h;
		w -= rate * dw;
		b -= rate * db;

		printf("loss = %f, w = %f, b = %f\n", loss(w, b), w, b);
	}

	printf("-------------------------\n");
	printf("w = %f, b = %f\n", w, b);

	return 0;
}
