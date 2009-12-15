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
	"runtime"
)

var n = flag.Int("n", 2000, "count")
var nCPU = flag.Int("ncpu", 4, "number of cpus")

func evalA(i, j int) float64 { return 1 / float64(((i+j)*(i+j+1)/2 + i + 1)) }

type Vec []float64

func (v Vec) Times(i, n int, u Vec, c chan int) {
	for ; i < n; i++ {
		v[i] = 0
		for j := 0; j < len(u); j++ {
			v[i] += evalA(i, j) * u[j]
		}
	}
	c <- 1
}

func (v Vec) TimesTransp(i, n int, u Vec, c chan int) {
	for ; i < n; i++ {
		v[i] = 0
		for j := 0; j < len(u); j++ {
			v[i] += evalA(j, i) * u[j]
		}
	}
	c <- 1
}

func wait(c chan int) {
	for i := 0; i < *nCPU; i++ {
		<-c
	}
}

func (v Vec) ATimesTransp(u Vec) {
	x := make(Vec, len(u))
	c := make(chan int, *nCPU)
	for i := 0; i < *nCPU; i++ {
		go x.Times(i*len(v) / *nCPU, (i+1)*len(v) / *nCPU, u, c)
	}
	wait(c)
	for i := 0; i < *nCPU; i++ {
		go v.TimesTransp(i*len(v) / *nCPU, (i+1)*len(v) / *nCPU, x, c)
	}
	wait(c)
}

func main() {
	flag.Parse()
	runtime.GOMAXPROCS(*nCPU)
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
