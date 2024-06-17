NAME=twice

all: $(SRC)
	clang -Wall -Wextra $(NAME).c
