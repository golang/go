// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package contraire

import "math/rand"

// chooseUniform selects an alternative from alts uniformly.
func chooseUniform[T any](alts []T) T {
	return alts[rand.Intn(len(alts))]
}

// alt is an alternative in a cumulative distribution function (CDF).
type alt[T any] struct {
	p int // 0 <= p <= 100
	f func() T
}

const checkCDF = false

// chooseCDF selects a single alternative by observing a uniform random variable
// and comparing it to the CDF of the given alternatives.
func chooseCDF[T any](alts ...alt[T]) T {
	if checkCDF {
		verify(alts)
	}
	sum := 0
	choice := rand.Intn(100)
	for _, alt := range alts {
		sum += alt.p
		if choice < sum {
			return alt.f()
		}
	}
	panic("alternative probabilities sum to less than 100")
}

// verify checks that the given alternatives are suitable for chooseCDF.
func verify[T any](alts []alt[T]) {
	sum := 0
	for _, alt := range alts {
		sum += alt.p
	}
	if sum != 100 {
		panic("alternative probabilities must sum to 100")
	}
}

// oneToN picks a number uniformly on [1, n].
func oneToN(n int) int {
	return rand.Intn(n) + 1
}
