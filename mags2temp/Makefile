all: bin/mags2temp

bin/mags2temp: bin src/main.c src/BufferedWriter.c src/BufferedWriter.h src/CieXyz.c src/CieXyz.h src/Filters.c src/Filters.h src/FitUtils.c src/FitUtils.h src/MagUtils.c src/MagUtils.h src/PlanckUtils.c src/PlanckUtils.h
	$(CC) src/main.c src/BufferedWriter.c src/CieXyz.c src/Filters.c src/FitUtils.c src/MagUtils.c src/PlanckUtils.c -o $@ -std=c99 -Wall -Wextra -lgsl -lgslcblas -lm

bin:
	mkdir -p bin

clean:
	rm -f bin/mags2temp
