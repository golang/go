// run -gcflags=-G=3

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type I interface {
	[]byte
}

func F[T I]() {
	var t T
	explodes(t)
}

func explodes(b []byte) {}

func main() {

}
