// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _() {
	a, b, b /* ERROR "b repeated on left side of :=" */ := 1, 2, 3
	_ = a
	_ = b
}

func _() {
	a, _, _ := 1, 2, 3 // multiple _'s ok
	_ = a
}

func _() {
	var b int
	a, b, b /* ERROR "b repeated on left side of :=" */ := 1, 2, 3
	_ = a
	_ = b
}

func _() {
	var a []int
	a /* ERRORx `non-name .* on left side of :=` */ [0], b := 1, 2
	_ = a
	_ = b
}

func _() {
	var a int
	a, a /* ERROR "a repeated on left side of :=" */ := 1, 2
	_ = a
}

func _() {
	var a, b int
	a, b := /* ERROR "no new variables on left side of :=" */ 1, 2
	_ = a
	_ = b
}
