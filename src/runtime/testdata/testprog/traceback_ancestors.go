// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"runtime"
)

func init() {
	register("TracebackAncestors", TracebackAncestors)
}

const numGoroutines = 3
const numFrames = 2

func TracebackAncestors() {
	w := make(chan struct{})
	recurseThenCallGo(w, numGoroutines, numFrames)
	<-w
	printStack()
	close(w)
}

func printStack() {
	buf := make([]byte, 1024)
	for {
		n := runtime.Stack(buf, true)
		if n < len(buf) {
			fmt.Print(string(buf[:n]))
			return
		}
		buf = make([]byte, 2*len(buf))
	}
}

func recurseThenCallGo(w chan struct{}, frames int, goroutines int) {
	if frames == 0 {
		// Signal to TracebackAncestors that we are done recursing and starting goroutines.
		w <- struct{}{}
		<-w
		return
	}
	if goroutines == 0 {
		// Start the next goroutine now that there are no more recursions left
		// for this current goroutine.
		go recurseThenCallGo(w, frames-1, numFrames)
		return
	}
	recurseThenCallGo(w, frames, goroutines-1)
}
