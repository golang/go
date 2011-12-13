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
 * The Computer Language Benchmarks Game
 * http://shootout.alioth.debian.org/
 *
 * contributed by The Go Authors.
 * Based on fannkuch.c by Heiner Marxen
 */

package main

import (
	"flag"
	"fmt"
)

var n = flag.Int("n", 7, "count")

func fannkuch(n int) int {
	if n < 1 {
		return 0
	}

	n1 := n - 1
	perm := make([]int, n)
	perm1 := make([]int, n)
	count := make([]int, n)

	for i := 0; i < n; i++ {
		perm1[i] = i // initial (trivial) permutation
	}

	r := n
	didpr := 0
	flipsMax := 0
	for {
		if didpr < 30 {
			for i := 0; i < n; i++ {
				fmt.Printf("%d", 1+perm1[i])
			}
			fmt.Printf("\n")
			didpr++
		}
		for ; r != 1; r-- {
			count[r-1] = r
		}

		if perm1[0] != 0 && perm1[n1] != n1 {
			flips := 0
			for i := 1; i < n; i++ { // perm = perm1
				perm[i] = perm1[i]
			}
			k := perm1[0] // cache perm[0] in k
			for {         // k!=0 ==> k>0
				for i, j := 1, k-1; i < j; i, j = i+1, j-1 {
					perm[i], perm[j] = perm[j], perm[i]
				}
				flips++
				// Now exchange k (caching perm[0]) and perm[k]... with care!
				j := perm[k]
				perm[k] = k
				k = j
				if k == 0 {
					break
				}
			}
			if flipsMax < flips {
				flipsMax = flips
			}
		}

		for ; r < n; r++ {
			// rotate down perm[0..r] by one
			perm0 := perm1[0]
			for i := 0; i < r; i++ {
				perm1[i] = perm1[i+1]
			}
			perm1[r] = perm0
			count[r]--
			if count[r] > 0 {
				break
			}
		}
		if r == n {
			return flipsMax
		}
	}
	return 0
}

func main() {
	flag.Parse()
	fmt.Printf("Pfannkuchen(%d) = %d\n", *n, fannkuch(*n))
}
