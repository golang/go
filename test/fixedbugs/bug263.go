// run

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	data := make(map[int]string, 1)
	data[0] = "hello, "
	data[0] += "world!"
	if data[0] != "hello, world!" {
		panic("BUG: " + data[0])
	}
}
