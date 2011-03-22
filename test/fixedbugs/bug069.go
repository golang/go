// $G $D/$F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	c := make(chan int);
	ok := false;
	var i int;
	
	i, ok = <-c;  // works
	_, _ = i, ok;
	
	ca := new([2]chan int);
	i, ok = <-(ca[0]);  // fails: c.go:11: bad shape across assignment - cr=1 cl=2
	_, _ = i, ok;
}
