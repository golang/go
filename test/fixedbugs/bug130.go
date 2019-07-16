// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "os"

type I interface { send(chan <- int) }

type S struct { v int }
func (p *S) send(c chan <- int) { c <- p.v }

func main() {
	s := S{0};
	var i I = &s;
	c := make(chan int);
	go i.send(c);
	os.Exit(<-c);
}
