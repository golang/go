// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"log"
	"runtime"

	"./mysync"
)

func main() {
	var wg mysync.WaitGroup
	wg.Done()
	ci := runtime.CallersFrames(wg.Callers)
	frames := make([]runtime.Frame, 0, 4)
	for {
		frame, more := ci.Next()
		frames = append(frames, frame)
		if !more {
			break
		}
	}
	expecting := []string{
		"test/mysync.(*WaitGroup).Add",
		"test/mysync.(*WaitGroup).Done",
	}
	for i := 0; i < 2; i++ {
		if frames[i].Function != expecting[i] {
			log.Fatalf("frame %d: got %s, want %s", i, frames[i].Function, expecting[i])
		}
	}
}
