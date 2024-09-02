// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"runtime"
	"runtime/mainthread"
	"sync"
)

func MainThread() {
	var wg sync.WaitGroup
	runtime.LockOSThread()
	wg.Add(1)
	go func() {
		defer wg.Done()
		mainthread.Do(func() { println("Ok") })
	}()
	<-mainthread.Waiting()
	mainthread.Yield()
	wg.Wait()
}

func init() {
	register("MainThread", func() {
		println("expect: Ok")
		MainThread()
	})
	register("MainThread2", func() {
		println("expect: hello,world")
		MainThread2()
	})
}

func MainThread2() {
	var wg sync.WaitGroup
	runtime.LockOSThread()
	wg.Add(1)
	go func() {
		defer wg.Done()
		mainthread.Do(func() {
			print("hello,")
			mainthread.Do(func() {
				print("world")
			})
		})
	}()
	<-mainthread.Waiting()
	mainthread.Yield()
	wg.Wait()
}
