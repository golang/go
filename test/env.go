// [ $GOOS != nacl ] || exit 0  # NaCl runner does not expose environment
// $G $F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import os "os"

func main() {
	ga, e0 := os.Getenverror("GOARCH")
	if e0 != nil {
		print("$GOARCH: ", e0.String(), "\n")
		os.Exit(1)
	}
	if ga != "amd64" && ga != "386" && ga != "arm" {
		print("$GOARCH=", ga, "\n")
		os.Exit(1)
	}
	xxx, e1 := os.Getenverror("DOES_NOT_EXIST")
	if e1 != os.ENOENV {
		print("$DOES_NOT_EXIST=", xxx, "; err = ", e1.String(), "\n")
		os.Exit(1)
	}
}
