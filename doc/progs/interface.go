// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains the code snippets included in "The Laws of Reflection."

package main

import (
	"bufio"
	"bytes"
	"io"
	"os"
)

type MyInt int

var i int
var j MyInt

// STOP OMIT

// Reader is the interface that wraps the basic Read method.
type Reader interface {
	Read(p []byte) (n int, err error)
}

// Writer is the interface that wraps the basic Write method.
type Writer interface {
	Write(p []byte) (n int, err error)
}

// STOP OMIT

func readers() { // OMIT
	var r io.Reader
	r = os.Stdin
	r = bufio.NewReader(r)
	r = new(bytes.Buffer)
	// and so on
	// STOP OMIT
}

func typeAssertions() (interface{}, error) { // OMIT
	var r io.Reader
	tty, err := os.OpenFile("/dev/tty", os.O_RDWR, 0)
	if err != nil {
		return nil, err
	}
	r = tty
	// STOP OMIT
	var w io.Writer
	w = r.(io.Writer)
	// STOP OMIT
	var empty interface{}
	empty = w
	// STOP OMIT
	return empty, err
}

func main() {
}
