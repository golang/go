// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "time"

func main() {
	var count int
	c := make(chan byte)
	go func(c chan byte) {
		<-c
		count++
		time.Sleep(1000000)
		count++
		<-c
	}(c)
	c <- 1
	c <- 2
	if count != 2 {
		panic("synchronous send did not wait")
	}
}
