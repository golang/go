// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var G int

func main() {
	if G != 101 {
		println("not 101")
	} else {
		println("well now that's interesting")
	}
}
