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
TODO(rsc): delete this comment
TODO(rsc): move to test/bench once package "big" is ready
on r45:

make clean
make install
goc pidigits.go
ulimit -v 1000000	# 1GB
LD_LIBRARY_PATH=$GOROOT/pkg/linux_amd64 6.out
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
	//"big";
	big "gmp";
	"flag";
	"fmt";
	"runtime";
)

var n = flag.Int("n", 27, "number of digits");
var silent = flag.Bool("s", false, "don't print result");

var (
	tmp1 = big.NewInt(0);
	tmp2 = big.NewInt(0);
	numer = big.NewInt(1);
	accum = big.NewInt(0);
	denom = big.NewInt(1);
	ten = big.NewInt(10);
)

func extract_digit() int64 {
	if big.CmpInt(numer, accum) > 0 {
		return -1;
	}

	// Compute (numer * 3 + accum) / denom
	tmp1.Lsh(numer, 1).Add(tmp1, numer).Add(tmp1, accum);
	big.DivModInt(tmp1, tmp2, tmp1, denom);

	// Now, if (numer * 4 + accum) % denom...
	tmp2.Add(tmp2, numer);

	// ... is normalized, then the two divisions have the same result.
	if big.CmpInt(tmp2, denom) >= 0 {
		return -1;
	}

	return tmp1.Int64();
}

func next_term(k int64) {
	y2 := k*2 + 1;

	accum.Add(accum, tmp1.Lsh(numer, 1));
	accum.Mul(accum, tmp1.SetInt64(y2));
	numer.Mul(numer, tmp1.SetInt64(k));
	denom.Mul(denom, tmp1.SetInt64(y2));
}

func eliminate_digit(d int64) {
	accum.Sub(accum, tmp1.Mul(denom, tmp1.SetInt64(d)));
	accum.Mul(accum, ten);
	numer.Mul(numer, ten);
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

	fmt.Printf("%d calls; %d %d %d\n", runtime.Cgocalls(), numer.Len(), accum.Len(), denom.Len());
}
