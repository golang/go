// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"runtime"
	"strings"
)

func init() {
	register("TracebackAncestors", TracebackAncestors)
}

const numGoroutines = 3
const numFrames = 2

func TracebackAncestors() {
	w := make(chan struct{})
	recurseThenCallGo(w, numGoroutines, numFrames, true)
	<-w
	printStack()
	close(w)
}

var ignoreGoroutines = make(map[string]bool)

func printStack() {
	buf := make([]byte, 1024)
	for {
		n := runtime.Stack(buf, true)
		if n < len(buf) {
			tb := string(buf[:n])

			// Delete any ignored goroutines, if present.
			pos := 0
			for pos < len(tb) {
				next := pos + strings.Index(tb[pos:], "\n\n")
				if next < pos {
					next = len(tb)
				} else {
					next += len("\n\n")
				}

				if strings.HasPrefix(tb[pos:], "goroutine ") {
					id := tb[pos+len("goroutine "):]
					id = id[:strings.IndexByte(id, ' ')]
					if ignoreGoroutines[id] {
						tb = tb[:pos] + tb[next:]
						next = pos
					}
				}
				pos = next
			}

			fmt.Print(tb)
			return
		}
		buf = make([]byte, 2*len(buf))
	}
}

func recurseThenCallGo(w chan struct{}, frames int, goroutines int, main bool) {
	if frames == 0 {
		// Signal to TracebackAncestors that we are done recursing and starting goroutines.
		w <- struct{}{}
		<-w
		return
	}
	if goroutines == 0 {
		// Record which goroutine this is so we can ignore it
		// in the traceback if it hasn't finished exiting by
		// the time we printStack.
		if !main {
			ignoreGoroutines[goroutineID()] = true
		}

		// Start the next goroutine now that there are no more recursions left
		// for this current goroutine.
		go recurseThenCallGo(w, frames-1, numFrames, false)
		return
	}
	recurseThenCallGo(w, frames, goroutines-1, main)
}

func goroutineID() string {
	buf := make([]byte, 128)
	runtime.Stack(buf, false)
	const prefix = "goroutine "
	if !bytes.HasPrefix(buf, []byte(prefix)) {
		panic(fmt.Sprintf("expected %q at beginning of traceback:\n%s", prefix, buf))
	}
	buf = buf[len(prefix):]
	n := bytes.IndexByte(buf, ' ')
	return string(buf[:n])
}
