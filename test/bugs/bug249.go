// errchk $G $D/$F.go

// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var c1 chan<- chan int
var c2 chan<- (chan int) // same type as c1 according to gccgo, gofmt
var c3 chan (<-chan int) // same type as c1 according to 6g

func main() {
	c1 = c2 // this should be ok, bug 6g doesn't accept it
	c1 = c3 // ERROR "chan"
}

/*
Channel types are parsed differently by 6g then by gccgo and gofmt.
The channel type specification ( http://golang.org/doc/go_spec.html#Channel_types )
says that a channel type is either

	chan ElementType
	chan <- ElementType
	<-chan ElementType

which indicates that the <- binds to the chan token (not to the ElementType).
So:

chan <- chan int

should be parsed as

chan<- (chan int)

Both gccgo and gofmt adhere to this, while 6g parses this as

chan (<-chan int)
*/
