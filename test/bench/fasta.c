/*
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.

    * Neither the name of "The Computer Language Benchmarks Game" nor the
    name of "The Computer Language Shootout Benchmarks" nor the names of
    its contributors may be used to endorse or promote products derived
    from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

/*
 * http://shootout.alioth.debian.org/u32q/benchmark.php?test=fasta&lang=gcc&id=4
*/
/* The Computer Language Benchmarks Game
 * http://shootout.alioth.debian.org/
 * Contributed by Joern Inge Vestgaarden
 * Modified by Jorge Peixoto de Morais Neto
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <err.h>

#define WIDTH 60
#define MIN(a,b) ((a) <= (b) ? (a) : (b))
#define NELEMENTS(x) (sizeof (x) / sizeof ((x)[0]))

typedef struct {
    float p;
    char c;
} aminoacid_t;

static inline float myrandom (float max) {
    unsigned long const IM = 139968;
    unsigned long const IA = 3877;
    unsigned long const IC = 29573;
    static unsigned long last = 42;
    last = (last * IA + IC) % IM;
    /*Integer to float conversions are faster if the integer is signed*/
    return max * (long) last / IM;
}

static inline void accumulate_probabilities (aminoacid_t *genelist, size_t len) {
    float cp = 0.0;
    size_t i;
    for (i = 0; i < len; i++) {
        cp += genelist[i].p;
        genelist[i].p = cp;
    }
}

/* This function prints the characters of the string s. When it */
/* reaches the end of the string, it goes back to the beginning */
/* It stops when the total number of characters printed is count. */
/* Between each WIDTH consecutive characters it prints a newline */
/* This function assumes that WIDTH <= strlen (s) + 1 */
static void repeat_fasta (char const *s, size_t count) {
    size_t pos = 0;
    size_t len = strlen (s);
    char *s2 = malloc (len + WIDTH);
    memcpy (s2, s, len);
    memcpy (s2 + len, s, WIDTH);
    do {
     	size_t line = MIN(WIDTH, count);
     	fwrite (s2 + pos,1,line,stdout);
     	putchar_unlocked ('\n');
     	pos += line;
     	if (pos >= len) pos -= len;
     	count -= line;
    } while (count);
    free (s2);
}

/* This function takes a pointer to the first element of an array */
/* Each element of the array is a struct with a character and */
/* a float number p between 0 and 1. */
/* The function generates a random float number r and */
/* finds the first array element such that p >= r. */
/* This is a weighted random selection. */
/* The function then prints the character of the array element. */
/* This is done count times. */
/* Between each WIDTH consecutive characters, the function prints a newline */
static void random_fasta (aminoacid_t const *genelist, size_t count) {
    do {
	size_t line = MIN(WIDTH, count);
	size_t pos = 0;
	char buf[WIDTH + 1];
	do {
	    float r = myrandom (1.0);
	    size_t i = 0;
	    while (genelist[i].p < r)
		++i; /* Linear search */
	    buf[pos++] = genelist[i].c;
	} while (pos < line);
	buf[line] = '\n';
	fwrite (buf, 1, line + 1, stdout);
	count -= line;
    } while (count);
}

int main (int argc, char **argv) {
    size_t n;
    if (argc > 1) {
	char const *arg = argv[1];
 	char *tail;
 	n = strtoul (arg, &tail, 0);
 	if (tail == arg)
	    errx (1, "Could not convert \"%s\" to an unsigned long integer", arg);
    } else n = 1000;

    static aminoacid_t iub[] = {
	{ 0.27, 'a' },
	{ 0.12, 'c' },
	{ 0.12, 'g' },
	{ 0.27, 't' },
	{ 0.02, 'B' },
	{ 0.02, 'D' },
	{ 0.02, 'H' },
	{ 0.02, 'K' },
	{ 0.02, 'M' },
	{ 0.02, 'N' },
	{ 0.02, 'R' },
	{ 0.02, 'S' },
	{ 0.02, 'V' },
	{ 0.02, 'W' },
	{ 0.02, 'Y' }};

    static aminoacid_t homosapiens[] = {
	{ 0.3029549426680, 'a' },
	{ 0.1979883004921, 'c' },
	{ 0.1975473066391, 'g' },
	{ 0.3015094502008, 't' }};

    accumulate_probabilities (iub, NELEMENTS(iub));
    accumulate_probabilities (homosapiens, NELEMENTS(homosapiens));

    static char const *const alu ="\
GGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGG\
GAGGCCGAGGCGGGCGGATCACCTGAGGTCAGGAGTTCGAGA\
CCAGCCTGGCCAACATGGTGAAACCCCGTCTCTACTAAAAAT\
ACAAAAATTAGCCGGGCGTGGTGGCGCGCGCCTGTAATCCCA\
GCTACTCGGGAGGCTGAGGCAGGAGAATCGCTTGAACCCGGG\
AGGCGGAGGTTGCAGTGAGCCGAGATCGCGCCACTGCACTCC\
AGCCTGGGCGACAGAGCGAGACTCCGTCTCAAAAA";

    fputs (">ONE Homo sapiens alu\n", stdout);
    repeat_fasta (alu, 2 * n);
    fputs (">TWO IUB ambiguity codes\n", stdout);
    random_fasta (iub, 3 * n);
    fputs (">THREE Homo sapiens frequency\n", stdout);
    random_fasta (homosapiens, 5 * n);
    return 0;
}
