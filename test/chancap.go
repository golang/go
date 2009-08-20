// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	c := make(chan int, 10);
	if len(c) != 0 || cap(c) != 10 {
		panicln("chan len/cap ", len(c), cap(c), " want 0 10");
	}

	for i := 0; i < 3; i++ {
		c <- i;
	}
	if len(c) != 3 || cap(c) != 10 {
		panicln("chan len/cap ", len(c), cap(c), " want 3 10");
	}
	
	c = make(chan int);
	if len(c) != 0 || cap(c) != 0 {
		panicln("chan len/cap ", len(c), cap(c), " want 0 0");
	}
}

