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
 * http://shootout.alioth.debian.org/u32/program.php?test=fasta&lang=gcc&id=3
 */

/*  The Computer Language Benchmarks Game
 *  http://shootout.alioth.debian.org/
 *
 *  contributed by Petr Prokhorenkov
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// not available on OS X 
#define fwrite_unlocked fwrite
#define fputc_unlocked fputc
#define fputs_unlocked fputs

#define ARRAY_SIZE(a) (sizeof(a)/sizeof(a[0]))
#define unlikely(x) __builtin_expect((x), 0)

#define IM 139968
#define IA 3877
#define IC 29573

#define LINE_LEN 60
#define LOOKUP_SIZE 4096
#define LOOKUP_SCALE ((float)(LOOKUP_SIZE - 1))

typedef unsigned random_t;

void
random_init(random_t *random) {
    *random = 42;
}

// Special version with result rescaled to LOOKUP_SCALE.
static inline
float
random_next_lookup(random_t *random) {
    *random = (*random*IA + IC)%IM;

    return (*random)*(LOOKUP_SCALE/IM);
}

struct amino_acid {
   char sym;
   float prob;
   float cprob_lookup;
};

void
repeat(const char *alu, const char *title, int n) {
    int len = strlen(alu);
    char buffer[len + LINE_LEN];
    int pos = 0;

    memcpy(buffer, alu, len);
    memcpy(buffer + len, alu, LINE_LEN);

    fputs_unlocked(title, stdout);
    while (n > 0) {
        int bytes = n > LINE_LEN ? LINE_LEN : n;

        fwrite_unlocked(buffer + pos, bytes, 1, stdout);
        pos += bytes;
        if (pos > len) {
            pos -= len;
        }
        fputc_unlocked('\n', stdout);
        n -= bytes;
    }
}

/*
 * Lookup table contains mapping from real values to cumulative
 * probabilities. Careful selection of table size allows lookup
 * virtually in constant time.
 *
 * All cumulative probabilities are rescaled to LOOKUP_SCALE,
 * this allows to save one multiplication operation on each iteration
 * in randomize().
 */

void *
fill_lookup(struct amino_acid **lookup, struct amino_acid *amino_acid, int amino_acid_size) {
    float p = 0;
    int i, j;

    for (i = 0; i < amino_acid_size; i++) {
        p += amino_acid[i].prob;
        amino_acid[i].cprob_lookup = p*LOOKUP_SCALE;
    }

    // Prevent rounding error.
    amino_acid[amino_acid_size - 1].cprob_lookup = LOOKUP_SIZE - 1;

    for (i = 0, j = 0; i < LOOKUP_SIZE; i++) {
        while (amino_acid[j].cprob_lookup < i) {
            j++;
        }
        lookup[i] = &amino_acid[j];
    }

    return 0;
}

void
randomize(struct amino_acid *amino_acid, int amino_acid_size,
        const char *title, int n, random_t *rand) {
    struct amino_acid *lookup[LOOKUP_SIZE];
    char line_buffer[LINE_LEN + 1];
    int i, j;

    line_buffer[LINE_LEN] = '\n';

    fill_lookup(lookup, amino_acid, amino_acid_size);

    fputs_unlocked(title, stdout);

    for (i = 0, j = 0; i < n; i++, j++) {
        if (j == LINE_LEN) {
            fwrite_unlocked(line_buffer, LINE_LEN + 1, 1, stdout);
            j = 0;
        }

        float r = random_next_lookup(rand);
        struct amino_acid *u = lookup[(short)r];
        while (unlikely(u->cprob_lookup < r)) {
            ++u;
        }
        line_buffer[j] = u->sym;
    }
    line_buffer[j] = '\n';
    fwrite_unlocked(line_buffer, j + 1, 1, stdout);
}

struct amino_acid amino_acid[] = {
   { 'a', 0.27 },
   { 'c', 0.12 },
   { 'g', 0.12 },
   { 't', 0.27 },

   { 'B', 0.02 },
   { 'D', 0.02 },
   { 'H', 0.02 },
   { 'K', 0.02 },
   { 'M', 0.02 },
   { 'N', 0.02 },
   { 'R', 0.02 },
   { 'S', 0.02 },
   { 'V', 0.02 },
   { 'W', 0.02 },
   { 'Y', 0.02 },
};

struct amino_acid homo_sapiens[] = {
   { 'a', 0.3029549426680 },
   { 'c', 0.1979883004921 },
   { 'g', 0.1975473066391 },
   { 't', 0.3015094502008 },
};

static const char alu[] =
   "GGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTG"
   "GGAGGCCGAGGCGGGCGGATCACCTGAGGTCAGGAGTTCGA"
   "GACCAGCCTGGCCAACATGGTGAAACCCCGTCTCTACTAAA"
   "AATACAAAAATTAGCCGGGCGTGGTGGCGCGCGCCTGTAAT"
   "CCCAGCTACTCGGGAGGCTGAGGCAGGAGAATCGCTTGAAC"
   "CCGGGAGGCGGAGGTTGCAGTGAGCCGAGATCGCGCCACTG"
   "CACTCCAGCCTGGGCGACAGAGCGAGACTCCGTCTCAAAAA";

int
main(int argc, const char **argv) {
    int n = argc > 1 ? atoi( argv[1] ) : 512;
    random_t rand;

    random_init(&rand);

    repeat(alu, ">ONE Homo sapiens alu\n", n*2);
    randomize(amino_acid, ARRAY_SIZE(amino_acid),
            ">TWO IUB ambiguity codes\n", n*3, &rand);
    randomize(homo_sapiens, ARRAY_SIZE(homo_sapiens),
            ">THREE Homo sapiens frequency\n", n*5, &rand);

    return 0;
}