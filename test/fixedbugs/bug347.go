// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"runtime"
	"strings"
)

var t *struct {
	c chan int
}

var c chan int

func f() {
	select {
	case <-t.c:  // THIS IS LINE 22
		break
	case <-c:
		break
	}
}

func main() {
	defer func() {
		recover()
		for i := 0;; i++ {
			pc, file, line, ok := runtime.Caller(i)
			if !ok {
				print("BUG: bug347: cannot find caller\n")
				return
			}
			if !strings.Contains(file, "bug347.go") || runtime.FuncForPC(pc).Name() != "main.f" {
				// walk past runtime frames
				continue
			}
			if line != 22 {
				print("BUG: bug347: panic at ", file, ":", line, " in ", runtime.FuncForPC(pc).Name(), "\n")
			}
			return
		}
	}()
	f()
}
