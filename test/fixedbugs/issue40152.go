// run

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Gccgo mishandles converting an untyped boolean to an interface type.

package main

func t(args ...interface{}) bool {
        x := true
        return x == args[0]
}

func main() {
	r := t("x" == "x" && "y" == "y")
	if !r {
		panic(r)
	}
}
