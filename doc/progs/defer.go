// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains the code snippets included in "Defer, Panic, and Recover."

package main

import (
	"fmt"
	"io"
	"os"
)

func a() {
	i := 0
	defer fmt.Println(i)
	i++
	return
}

// STOP OMIT

func b() {
	for i := 0; i < 4; i++ {
		defer fmt.Print(i)
	}
}

// STOP OMIT

func c() (i int) {
	defer func() { i++ }()
	return 1
}

// STOP OMIT

// Initial version.
func CopyFile(dstName, srcName string) (written int64, err error) {
	src, err := os.Open(srcName)
	if err != nil {
		return
	}

	dst, err := os.Create(dstName)
	if err != nil {
		return
	}

	written, err = io.Copy(dst, src)
	dst.Close()
	src.Close()
	return
}

// STOP OMIT

func main() {
	a()
	b()
	fmt.Println()
	fmt.Println(c())
}
