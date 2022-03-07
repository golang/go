// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	var a, b foo
	bar(a, b)
}

type foo int

func (a foo) less(b foo) bool {
	return a < b
}

type lesser[T any] interface {
	less(T) bool
	comparable
}

func bar[T lesser[T]](a, b T) {
	a.less(b)
}
