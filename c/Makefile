CC = cc
CFLAGS = -O2 -std=c11 -Wall
LDFLAGS = -lm

microgpt: microgpt.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f microgpt

.PHONY: clean
