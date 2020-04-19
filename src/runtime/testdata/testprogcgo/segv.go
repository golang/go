// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !plan9,!windows

package main

// static void nop() {}
import "C"

import (
	"sync"
	"syscall"
)

func init() {
	register("Segv", Segv)
	register("SegvInCgo", SegvInCgo)
}

var Sum int

func Segv() {
	c := make(chan bool)
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		close(c)
		for i := 0; i < 10000; i++ {
			Sum += i
		}
	}()

	<-c

	syscall.Kill(syscall.Getpid(), syscall.SIGSEGV)

	wg.Wait()
}

func SegvInCgo() {
	c := make(chan bool)
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		close(c)
		for i := 0; i < 10000; i++ {
			C.nop()
		}
	}()

	<-c

	syscall.Kill(syscall.Getpid(), syscall.SIGSEGV)

	wg.Wait()
}
