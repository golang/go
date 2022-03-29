// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !plan9 && !windows
// +build !plan9,!windows

package main

// static void nop() {}
import "C"

import (
	"syscall"
	"time"
)

func init() {
	register("Segv", Segv)
	register("SegvInCgo", SegvInCgo)
}

var Sum int

func Segv() {
	c := make(chan bool)
	go func() {
		close(c)
		for i := 0; ; i++ {
			Sum += i
		}
	}()

	<-c

	syscall.Kill(syscall.Getpid(), syscall.SIGSEGV)

	// Give the OS time to deliver the signal.
	time.Sleep(time.Second)
}

func SegvInCgo() {
	c := make(chan bool)
	go func() {
		close(c)
		for {
			C.nop()
		}
	}()

	<-c

	syscall.Kill(syscall.Getpid(), syscall.SIGSEGV)

	// Give the OS time to deliver the signal.
	time.Sleep(time.Second)
}
