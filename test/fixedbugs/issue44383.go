// compile

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 44383: gofrontend internal compiler error

package main

func main() {
	var b1, b2 byte
	f := func() int {
		var m map[byte]int
		return m[b1/b2]
	}
	f()
}
