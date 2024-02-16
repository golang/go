// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// #include <unistd.h>
// static void nop() {}
import "C"

import "syscall"

func init() {
	register("TgkillSegvInCgo", TgkillSegvInCgo)
}

func TgkillSegvInCgo() {
	c := make(chan bool)
	go func() {
		close(c)
		for {
			C.nop()
		}
	}()

	<-c

	syscall.Tgkill(syscall.Getpid(), syscall.Gettid(), syscall.SIGSEGV)

	// Wait for the OS to deliver the signal.
	C.pause()
}
