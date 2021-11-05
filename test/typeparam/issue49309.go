// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func genfunc[T any](f func(c T)) {
	var r T

	f(r)
}

func myfunc(c string) {
	test2(c)
}

//go:noinline
func test2(a interface{}) {
}

func main() {
	genfunc(myfunc)
}
