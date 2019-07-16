// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {	
	c := make(chan bool, 1);
	select {
	case _ = <-c:
		panic("BUG: recv should not");
	default:
	}
	c <- true;
	select {
	case _ = <-c:
	default:
		panic("BUG: recv should");
	}
}
