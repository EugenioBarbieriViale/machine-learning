NAME=ml

all: $(SRC)
	clang -Wall -Wextra $(NAME).c
