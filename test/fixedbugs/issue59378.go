// compile

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f() {
	F([]int{}, func(*int) bool { return true })
}

func F[S []E, E any](a S, fn func(*E) bool) {
	for _, v := range a {
		G(a, func(e E) bool { return fn(&v) })
	}
}

func G[E any](s []E, f func(E) bool) int {
	for i, v := range s {
		if f(v) {
			return i
		}
	}
	return -1
}
