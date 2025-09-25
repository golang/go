// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lcs

import (
	"fmt"
)

// For each D, vec[D] has length D+1,
// and the label for (D, k) is stored in vec[D][(D+k)/2].
type label struct {
	vec [][]int
}

// Temporary checking DO NOT COMMIT true TO PRODUCTION CODE
const debug = false

// debugging. check that the (d,k) pair is valid
// (that is, -d<=k<=d and d+k even)
func checkDK(D, k int) {
	if k >= -D && k <= D && (D+k)%2 == 0 {
		return
	}
	panic(fmt.Sprintf("out of range, d=%d,k=%d", D, k))
}

func (t *label) set(D, k, x int) {
	if debug {
		checkDK(D, k)
	}
	for len(t.vec) <= D {
		t.vec = append(t.vec, nil)
	}
	if t.vec[D] == nil {
		t.vec[D] = make([]int, D+1)
	}
	t.vec[D][(D+k)/2] = x // known that D+k is even
}

func (t *label) get(d, k int) int {
	if debug {
		checkDK(d, k)
	}
	return int(t.vec[d][(d+k)/2])
}

func newtriang(limit int) label {
	if limit < 100 {
		// Preallocate if limit is not large.
		return label{vec: make([][]int, limit)}
	}
	return label{}
}
