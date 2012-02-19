// run

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test for select: Issue 2075
// A bug in select corrupts channel queues of failed cases
// if there are multiple waiters on those channels and the
// select is the last in the queue. If further waits are made
// on the channel without draining it first then those waiters
// will never wake up. In the code below c1 is such a channel.

package main

func main() {
	c1 := make(chan bool)
	c2 := make(chan bool)
	c3 := make(chan bool)
	go func() { <-c1 }()
	go func() {
		select {
		case <-c1:
			panic("dummy")
		case <-c2:
			c3 <- true
		}
		<-c1
	}()
	go func() { c2 <- true }()
	<-c3
	c1 <- true
	c1 <- true
}
