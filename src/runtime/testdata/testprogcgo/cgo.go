// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

/*
void foo1(void) {}
void foo2(void* p) {}
*/
import "C"
import (
	"fmt"
	"os"
	"runtime"
	"strconv"
	"time"
	"unsafe"
)

func init() {
	register("CgoSignalDeadlock", CgoSignalDeadlock)
	register("CgoTraceback", CgoTraceback)
	register("CgoCheckBytes", CgoCheckBytes)
}

func CgoSignalDeadlock() {
	runtime.GOMAXPROCS(100)
	ping := make(chan bool)
	go func() {
		for i := 0; ; i++ {
			runtime.Gosched()
			select {
			case done := <-ping:
				if done {
					ping <- true
					return
				}
				ping <- true
			default:
			}
			func() {
				defer func() {
					recover()
				}()
				var s *string
				*s = ""
				fmt.Printf("continued after expected panic\n")
			}()
		}
	}()
	time.Sleep(time.Millisecond)
	start := time.Now()
	var times []time.Duration
	for i := 0; i < 64; i++ {
		go func() {
			runtime.LockOSThread()
			select {}
		}()
		go func() {
			runtime.LockOSThread()
			select {}
		}()
		time.Sleep(time.Millisecond)
		ping <- false
		select {
		case <-ping:
			times = append(times, time.Since(start))
		case <-time.After(time.Second):
			fmt.Printf("HANG 1 %v\n", times)
			return
		}
	}
	ping <- true
	select {
	case <-ping:
	case <-time.After(time.Second):
		fmt.Printf("HANG 2 %v\n", times)
		return
	}
	fmt.Printf("OK\n")
}

func CgoTraceback() {
	C.foo1()
	buf := make([]byte, 1)
	runtime.Stack(buf, true)
	fmt.Printf("OK\n")
}

func CgoCheckBytes() {
	try, _ := strconv.Atoi(os.Getenv("GO_CGOCHECKBYTES_TRY"))
	if try <= 0 {
		try = 1
	}
	b := make([]byte, 1e6*try)
	start := time.Now()
	for i := 0; i < 1e3*try; i++ {
		C.foo2(unsafe.Pointer(&b[0]))
		if time.Since(start) > time.Second {
			break
		}
	}
}
