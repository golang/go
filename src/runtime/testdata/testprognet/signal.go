// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !windows,!plan9

// This is in testprognet instead of testprog because testprog
// must not import anything (like net, but also like os/signal)
// that kicks off background goroutines during init.

package main

import (
	"os/signal"
	"syscall"
)

func init() {
	register("SignalIgnoreSIGTRAP", SignalIgnoreSIGTRAP)
}

func SignalIgnoreSIGTRAP() {
	signal.Ignore(syscall.SIGTRAP)
	syscall.Kill(syscall.Getpid(), syscall.SIGTRAP)
	println("OK")
}
