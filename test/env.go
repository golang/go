// $G $F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"os"
	"runtime"
)

func main() {
	ga, e0 := os.Getenverror("GOARCH")
	if e0 != nil {
		print("$GOARCH: ", e0.Error(), "\n")
		os.Exit(1)
	}
	if ga != runtime.GOARCH {
		print("$GOARCH=", ga, "!= runtime.GOARCH=", runtime.GOARCH, "\n")
		os.Exit(1)
	}
	xxx, e1 := os.Getenverror("DOES_NOT_EXIST")
	if e1 != os.ENOENV {
		print("$DOES_NOT_EXIST=", xxx, "; err = ", e1.Error(), "\n")
		os.Exit(1)
	}
}
