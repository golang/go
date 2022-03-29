// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that type substitution works and export/import works correctly even for a
// generic type that has multiple blank type params.

package main

import (
	"./a"
	"fmt"
)

func main() {
	var x a.T[int, a.Myint, string]
	fmt.Printf("%v\n", x)
}
