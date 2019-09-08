// skip

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"time"
)

func main() {
	// Make a big map.
	m := map[int]int{}
	for i := 0; i < 100000; i++ {
		m[i] = i
	}
	c := make(chan string)
	go func() {
		// Print the map.
		s := fmt.Sprintln(m)
		c <- s
	}()
	go func() {
		time.Sleep(1 * time.Millisecond)
		// Add an extra item to the map while iterating.
		m[-1] = -1
		c <- ""
	}()
	<-c
	<-c
}
