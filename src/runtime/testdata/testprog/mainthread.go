// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"runtime"
	"runtime/mainthread"
)

func MainThread() {
	go func() {
		mainthread.Do(func() { println("Ok") })
	}()
	<-mainthread.Waiting()
	mainthread.Yield()
}

func init() {
	runtime.LockOSThread()
	register("MainThread", func() {
		println("expect: Ok")
		MainThread()
	})
	register("MainThread2", func() {
		println("expect: hello,runtime: nested call mainthread.Do")
		MainThread2()
	})
}

func MainThread2() {
	defer func() {
		if err := recover(); err != nil {
			fmt.Print(err)
		}
	}()
	go func() {
		mainthread.Do(func() {
			print("hello,")
			mainthread.Do(func() {
				print("world")
			})
		})
	}()
	<-mainthread.Waiting()
	mainthread.Yield()
}
