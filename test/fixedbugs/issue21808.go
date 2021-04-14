// run

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Make sure println() prints a blank line.

package main

import "fmt"

func main() {
	fmt.Println("A")
	println()
	fmt.Println("B")
}
