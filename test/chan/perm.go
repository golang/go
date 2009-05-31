// errchk $G -e $D/$F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var (
	cr <-chan int;
	cs chan<- int;
	c chan int;
)

func main() {
	cr = c;		// ok
	cs = c;		// ok
	c = cr;		// ERROR "illegal types|incompatible"
	c = cs;		// ERROR "illegal types|incompatible"
	cr = cs;	// ERROR "illegal types|incompatible"
	cs = cr;	// ERROR "illegal types|incompatible"

	c <- 0;		// ok
	ok := c <- 0;	// ok
	<-c;		// ok
	x, ok := <-c;	// ok

	cr <- 0;	// ERROR "send"
	ok = cr <- 0;	// ERROR "send"
	<-cr;		// ok
	x, ok = <-cr;	// ok

	cs <- 0;	// ok
	ok = cs <- 0;	// ok
	<-cs;		// ERROR "receive"
	x, ok = <-cs;	// ERROR "receive"

	select {
	case c <- 0:	// ok
	case x := <-c:	// ok

	case cr <- 0:	// ERROR "send"
	case x := <-cr:	// ok

	case cs <- 0:	// ok;
	case x := <-cs:	// ERROR "receive"
	}
}
