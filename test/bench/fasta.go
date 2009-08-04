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

/* The Computer Language Benchmarks Game
 * http://shootout.alioth.debian.org/
 *
 * contributed by The Go Authors.
 * Based on C program by Joern Inge Vestgaarden
 * and Jorge Peixoto de Morais Neto.
 */

package main

import (
	"bufio";
	"bytes";
	"flag";
	"os";
	"strings";
)

var out *bufio.Writer

var n = flag.Int("n", 1000, "length of result")

const WIDTH = 60

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

type AminoAcid struct {
	p	float;
	c	byte;
}

var	lastrandom uint32 = 42

func myrandom() float {
	const (
		IM = 139968;
		IA = 3877;
		IC = 29573;
	)
	lastrandom = (lastrandom * IA + IC) % IM;
	// Integer to float conversions are faster if the integer is signed.
	return float(lastrandom) / IM;
}

func AccumulateProbabilities(genelist []AminoAcid) {
	cp := 0.0;
	for i := 0; i < len(genelist); i++ {
		cp += genelist[i].p;
		genelist[i].p = cp;
	}
}

/* This function prints the characters of the string s. When it */
/* reaches the end of the string, it goes back to the beginning */
/* It stops when the total number of characters printed is count. */
/* Between each WIDTH consecutive characters it prints a newline */
/* This function assumes that WIDTH <= strlen (s) + 1 */
func RepeatFasta(s []byte, count int) {
	pos := 0;
	s2 := make([]byte, len(s) + WIDTH);
	bytes.Copy(s2, s);
	bytes.Copy(s2[len(s):len(s2)], s);
	for {
		line := min(WIDTH, count);
		out.Write(s2[pos:pos+line]);
		out.WriteByte('\n');
		pos += line;
		if pos >= len(s) {
			pos -= len(s);
		}
		count -= line;
		if count <= 0 {
			break
		}
	}
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
func RandomFasta(genelist []AminoAcid, count int) {
	buf := make([]byte, WIDTH + 1);
	for {
		line := min(WIDTH, count);
		pos := 0;
		for {
			r := myrandom();
			var i int;
			for i = 0; genelist[i].p < r; i++ {
			}
			buf[pos] = genelist[i].c;
			pos++;
			if pos >= line {
				break
			}
		}
		buf[line] = '\n';
		out.Write(buf[0:line + 1]);
		count -= line;
		if count <= 0 {
			break
		}
	}
}

func main() {
	out = bufio.NewWriter(os.Stdout);
	defer out.Flush();

	flag.Parse();

	iub := []AminoAcid {
		AminoAcid{ 0.27, 'a' },
		AminoAcid{ 0.12, 'c' },
		AminoAcid{ 0.12, 'g' },
		AminoAcid{ 0.27, 't' },
		AminoAcid{ 0.02, 'B' },
		AminoAcid{ 0.02, 'D' },
		AminoAcid{ 0.02, 'H' },
		AminoAcid{ 0.02, 'K' },
		AminoAcid{ 0.02, 'M' },
		AminoAcid{ 0.02, 'N' },
		AminoAcid{ 0.02, 'R' },
		AminoAcid{ 0.02, 'S' },
		AminoAcid{ 0.02, 'V' },
		AminoAcid{ 0.02, 'W' },
		AminoAcid{ 0.02, 'Y' }
	};

	homosapiens := []AminoAcid {
		AminoAcid{ 0.3029549426680, 'a' },
		AminoAcid{ 0.1979883004921, 'c' },
		AminoAcid{ 0.1975473066391, 'g' },
		AminoAcid{ 0.3015094502008, 't' }
	};

	AccumulateProbabilities(iub);
	AccumulateProbabilities(homosapiens);

	alu := strings.Bytes(""
		"GGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGG"
		"GAGGCCGAGGCGGGCGGATCACCTGAGGTCAGGAGTTCGAGA"
		"CCAGCCTGGCCAACATGGTGAAACCCCGTCTCTACTAAAAAT"
		"ACAAAAATTAGCCGGGCGTGGTGGCGCGCGCCTGTAATCCCA"
		"GCTACTCGGGAGGCTGAGGCAGGAGAATCGCTTGAACCCGGG"
		"AGGCGGAGGTTGCAGTGAGCCGAGATCGCGCCACTGCACTCC"
		"AGCCTGGGCGACAGAGCGAGACTCCGTCTCAAAAA");

	out.WriteString(">ONE Homo sapiens alu\n");
	RepeatFasta(alu, 2 * *n);
	out.Flush();
	out.WriteString(">TWO IUB ambiguity codes\n");
	RandomFasta(iub, 3 * *n);
	out.Flush();
	out.WriteString(">THREE Homo sapiens frequency\n");
	RandomFasta(homosapiens, 5 * *n);
	out.Flush();
}
