Default:
	gcc main.c genann.c -g -Wall -Wextra -o main
	rm -r main.dSYM
clean:
	rm -f main
	rm -r main.dSYM