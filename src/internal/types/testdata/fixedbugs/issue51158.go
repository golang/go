// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

// Type checking the following code should not cause an infinite recursion.
func f[M map[K]int, K comparable](m M) {
        f(m)
}

// Equivalent code using mutual recursion.
func f1[M map[K]int, K comparable](m M) {
        f2(m)
}
func f2[M map[K]int, K comparable](m M) {
        f1(m)
}
