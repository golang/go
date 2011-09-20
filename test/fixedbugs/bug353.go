// errchk $G $D/$F.go

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// issue 2089 - internal compiler error

package main

import (
	"io"
	"os"
)

func echo(fd io.ReadWriterCloser) { // ERROR "undefined.*io.ReadWriterCloser"
	var buf [1024]byte
	for {
		n, err := fd.Read(buf)
		if err != nil {
			break
		}
		fd.Write(buf[0:n])
	}
}

func main() {
	fd, _ := os.Open("a.txt")
	echo(fd)
}
