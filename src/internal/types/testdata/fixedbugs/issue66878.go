// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _[T bool](ch chan T) {
	var _, _ T = <-ch
}

// offending code snippets from issue

func _[T ~bool](ch <-chan T) {
	var x, ok T = <-ch
	println(x, ok)
}

func _[T ~bool](m map[int]T) {
	var x, ok T = m[0]
	println(x, ok)
}
