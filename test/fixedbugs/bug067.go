// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var c chan int

func main() {
	c = make(chan int);
	go func() { print("ok\n"); c <- 0 } ();
	<-c
}
