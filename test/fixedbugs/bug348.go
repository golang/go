// run

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"runtime"
	"strings"
)

func f() {
	var x *string
	
	for _, i := range *x {  // THIS IS LINE 17
		println(i)
	}
}

func g() {
}

func main() {
	defer func() {
		for i := 0;; i++ {
			pc, file, line, ok := runtime.Caller(i)
			if !ok {
				print("BUG: bug348: cannot find caller\n")
				return
			}
			if !strings.Contains(file, "bug348.go") || runtime.FuncForPC(pc).Name() != "main.f" {
				// walk past runtime frames
				continue
			}
			if line != 17 {
				print("BUG: bug348: panic at ", file, ":", line, " in ", runtime.FuncForPC(pc).Name(), "\n")
				return
			}
			recover()
			return
		}
	}()
	f()
}
