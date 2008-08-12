// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// $G $F.go && $L $F.$A && ./$A.out readfile.go
// # This is some data we can recognize

package main

func main() {
	var s string
	var ok bool

	s, ok = sys.readfile("readfile.go");
	if !ok {
		print("couldn't readfile\n");
		sys.exit(1)
	}
	start_of_file :=
		"// $G $F.go && $L $F.$A && ./$A.out readfile.go\n" +
		"// # This is some data we can recognize\n" +
		"\n" +
		"package main\n";
	if s[0:102] != start_of_file {
		print("wrong data\n");
		sys.exit(1)
	}
}
