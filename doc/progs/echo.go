// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"os";
	"flag";  // command line option parser
)

var n_flag = flag.Bool("n", false, "don't print final newline")

const (
	kSpace = " ";
	kNewline = "\n";
)

func main() {
	flag.Parse();   // Scans the arg list and sets up flags
	var s string = "";
	for i := 0; i < flag.NArg(); i++ {
		if i > 0 {
			s += kSpace
		}
		s += flag.Arg(i)
	}
	if !*n_flag {
		s += kNewline
	}
	os.Stdout.WriteString(s);
}
