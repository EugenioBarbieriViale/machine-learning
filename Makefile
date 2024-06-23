NAME=xor

all: $(SRC)
	clang -o $(NAME) -Wall -Wextra $(NAME).c -lm
