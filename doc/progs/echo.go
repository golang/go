// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	OS "os";
	Flag "flag";
)

var n_flag = Flag.Bool("n", false, nil, "don't print final newline")

const (
	Space = " ";
	Newline = "\n";
)

func main() {
	Flag.Parse();   // Scans the arg list and sets up flags
	var s string = "";
	for i := 0; i < Flag.NArg(); i++ {
		if i > 0 {
			s += Space
		}
		s += Flag.Arg(i)
	}
	if !n_flag.BVal() {
		s += Newline
	}
	OS.Stdout.WriteString(s);
}
