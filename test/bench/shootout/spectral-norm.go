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
 * Based on spectral-norm.c by Sebastien Loisel
 */

package main

import (
	"flag"
	"fmt"
	"math"
)

var n = flag.Int("n", 2000, "count")

func evalA(i, j int) float64 { return 1 / float64(((i+j)*(i+j+1)/2 + i + 1)) }

type Vec []float64

func (v Vec) Times(u Vec) {
	for i := 0; i < len(v); i++ {
		v[i] = 0
		for j := 0; j < len(u); j++ {
			v[i] += evalA(i, j) * u[j]
		}
	}
}

func (v Vec) TimesTransp(u Vec) {
	for i := 0; i < len(v); i++ {
		v[i] = 0
		for j := 0; j < len(u); j++ {
			v[i] += evalA(j, i) * u[j]
		}
	}
}

func (v Vec) ATimesTransp(u Vec) {
	x := make(Vec, len(u))
	x.Times(u)
	v.TimesTransp(x)
}

func main() {
	flag.Parse()
	N := *n
	u := make(Vec, N)
	for i := 0; i < N; i++ {
		u[i] = 1
	}
	v := make(Vec, N)
	for i := 0; i < 10; i++ {
		v.ATimesTransp(u)
		u.ATimesTransp(v)
	}
	var vBv, vv float64
	for i := 0; i < N; i++ {
		vBv += u[i] * v[i]
		vv += v[i] * v[i]
	}
	fmt.Printf("%0.9f\n", math.Sqrt(vBv/vv))
}
