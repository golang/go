// compile

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package y

type symSet []int

//go:noinline
func (s symSet) len() (r int) {
	return 0
}

func f(m map[int]symSet) {
	var symSet []int
	for _, x := range symSet {
		m[x] = nil
	}
}
