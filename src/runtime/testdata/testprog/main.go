// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"os"
	"runtime"
	"strings"
)

var cmds = map[string]func(){}

func register(name string, f func()) {
	if cmds[name] != nil {
		panic("duplicate registration: " + name)
	}
	cmds[name] = f
}

func registerInit(name string, f func()) {
	if len(os.Args) >= 2 && os.Args[1] == name {
		f()
	}
}

func main() {
	if len(os.Args) < 2 {
		println("usage: " + os.Args[0] + " name-of-test")
		return
	}
	if !strings.Contains(os.Args[1], "MainThread") {
		// MainThread and MainThread2 need runtime.LockOSThread,
		// but will break MainGoroutineID.
		runtime.UnlockOSThread()
	}
	f := cmds[os.Args[1]]
	if f == nil {
		println("unknown function: " + os.Args[1])
		return
	}
	f()
}
