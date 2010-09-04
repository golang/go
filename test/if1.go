// $G $F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "os"

func main() {
	count := 7
	if one := 1; {
		count = count + one
	}
	if count != 8 {
		print(count, " should be 8\n")
		os.Exit(1)
	}
}
