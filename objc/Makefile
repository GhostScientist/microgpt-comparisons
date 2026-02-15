CC = clang
CFLAGS = -O2 -fobjc-arc -framework Foundation
LDFLAGS = -lm

microgpt: microgpt.m
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f microgpt

.PHONY: clean
