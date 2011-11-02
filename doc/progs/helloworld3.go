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
	f, err := file.Open("/does/not/exist")
	if f == nil {
		fmt.Printf("can't open file; err=%s\n", err.Error())
		os.Exit(1)
	}
}
