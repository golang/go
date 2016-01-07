// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !windows,!plan9,!nacl

package main

import "syscall"

func init() {
	register("SignalExitStatus", SignalExitStatus)
}

func SignalExitStatus() {
	syscall.Kill(syscall.Getpid(), syscall.SIGTERM)
}
