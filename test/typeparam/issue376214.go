// run -gcflags=-G=3

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func add[S ~string | ~[]byte](buf *[]byte, s S) {
	*buf = append(*buf, s...)
}

func main() {
	var buf []byte
	add(&buf, "foo")
	add(&buf, []byte("bar"))
	if string(buf) != "foobar" {
		panic("got " + string(buf))
	}
}
