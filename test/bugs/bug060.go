// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	m := new(map[int]int);
	m[0] = 0;
	m[0]++;
	if m[0] != 1 {
		print "map does not increment";
		sys.exit(1)
	}
}
