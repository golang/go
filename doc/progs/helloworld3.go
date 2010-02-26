// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"./file"
	"fmt"
	"os"
)

func main() {
	hello := []byte("hello, world\n")
	file.Stdout.Write(hello)
	file, err := file.Open("/does/not/exist",  0,  0)
	if file == nil {
		fmt.Printf("can't open file; err=%s\n",  err.String())
		os.Exit(1)
	}
}
