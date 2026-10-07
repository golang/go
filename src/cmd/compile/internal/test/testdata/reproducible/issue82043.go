// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type Regexp struct{ n int }

//go:noinline
func (re *Regexp) find(dstCap []int) []int { return append(dstCap, 0, re.n) }

func (re *Regexp) Find(b []byte) []byte {
	var dstCap [2]int
	a := re.find(dstCap[:0])
	if a == nil {
		return nil
	}
	return b[a[0]:a[1]:a[1]]
}

// The backing array of this literal is a noalg [2]int, sharing a descriptor
// with the [2]int above.
func (re *Regexp) allIndex(m []int) func(yield func([]int) bool) {
	return func(yield func([]int) bool) {
		yield([]int{m[0], m[1]})
	}
}
