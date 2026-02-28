// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package b

var slice = []int{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}

func F() ([]*int, []*int) {
	return g()
}

func g() ([]*int, []*int) {
	var s []*int
	var t []*int
	for i, j := range slice {
		s = append(s, &i)
		t = append(t, &j)
	}
	return s[:len(s)-1], t
}
