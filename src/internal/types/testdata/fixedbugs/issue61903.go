// -lang=go1.20

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type T[P any] interface{}

func f1[P any](T[P])    {}
func f2[P any](T[P], P) {}

func _() {
	var t T[int]
	f1(t)

	var s string
	f2(t, s /* ERROR "type string of s does not match inferred type int for P" */)
}
