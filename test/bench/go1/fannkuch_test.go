// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This benchmark, taken from the shootout, tests array indexing
// and array bounds elimination performance.

package go1

import "testing"

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

func BenchmarkFannkuch11(b *testing.B) {
	for i := 0; i < b.N; i++ {
		fannkuch(11)
	}
}
