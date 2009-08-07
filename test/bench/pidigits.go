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
 * based on pidigits.c (by Paolo Bonzini & Sean Bartlett,
 *                      modified by Michael Mellor)
 */

package main

import (
	"bignum";
	"flag";
	"fmt";
)

var n = flag.Int("n", 27, "number of digits");

// TODO for easier profiling, remove eventually
var silent = flag.Bool("s", false, "don't print result");

var (
	tmp1 *bignum.Integer;
	tmp2 *bignum.Integer;
	numer = bignum.Int(1);
	accum = bignum.Int(0);
	denom = bignum.Int(1);
)

func extract_digit() int64 {
	if numer.Cmp(accum) > 0 {
		return -1;
	}

	/* Compute (numer * 3 + accum) / denom */
	tmp1, tmp2 = numer.MulNat(bignum.Nat(3)).Add(accum).QuoRem(denom);

	/* Now, if (numer * 4 + accum) % denom... */
	tmp2 = tmp2.Add(numer);

	/* ... is normalized, then the two divisions have the same result.  */
	if tmp2.Cmp(denom) >= 0 {
		return -1;
	}

	return tmp1.Value();
}

func next_term(k int64) {
	y2 := k*2 + 1;

	tmp1 = numer.Shl(1);
	accum = accum.Add(tmp1);
	accum = accum.Mul1(y2);
	numer = numer.Mul1(k);
	denom = denom.Mul1(y2);
}

func eliminate_digit(d int64) {
	accum = accum.Sub(denom.Mul1(d));
	accum = accum.Mul1(10);
	numer = numer.Mul1(10);
}

func printf(s string, arg ...) {
	if !*silent {
		fmt.Printf(s, arg);
	}
}

func main() {
	flag.Parse();

	var m int;	// 0 <= m < 10
	for i, k := 0, int64(0); ; {
		d := int64(-1);
		for d < 0 {
			k++;
			next_term(k);
			d = extract_digit();
		}

		printf("%c", d + '0');

		i++;
		m = i%10;
		if m == 0 {
			printf("\t:%d\n", i);
		}
		if i >= *n {
			break;
		}
		eliminate_digit(d);
	}

	if m > 0 {
		printf("%s\t:%d\n", "          "[m : 10], *n);
	}
}
