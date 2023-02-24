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
			all := string(buf[:n])
			var saved string

			// Delete any ignored goroutines, if present.
			for all != "" {
				var g string
				g, all, _ = strings.Cut(all, "\n\n")

				if strings.HasPrefix(g, "goroutine ") {
					id, _, _ := strings.Cut(strings.TrimPrefix(g, "goroutine "), " ")
					if ignoreGoroutines[id] {
						continue
					}
				}
				if saved != "" {
					saved += "\n\n"
				}
				saved += g
			}

			fmt.Print(saved)
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
	prefix := []byte("goroutine ")
	var found bool
	if buf, found = bytes.CutPrefix(buf, prefix); !found {
		panic(fmt.Sprintf("expected %q at beginning of traceback:\n%s", prefix, buf))
	}
	id, _, _ := bytes.Cut(buf, []byte(" "))
	return string(id)
}
