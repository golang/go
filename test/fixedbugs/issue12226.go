// run

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

func main() {
	if []byte("foo")[0] == []byte("b")[0] {
		fmt.Println("BUG: \"foo\" and \"b\" appear to have the same first byte")
	}
}
