// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"internal/syscall/windows"
	"os"
	"runtime"
	"runtime/trace"
	"sync"
	"syscall"
	"time"
	"unsafe"
)

func init() {
	register("RaiseException", RaiseException)
	register("ZeroDivisionException", ZeroDivisionException)
	register("StackMemory", StackMemory)
	register("CallbackInAnotherThreadTrace", CallbackInAnotherThreadTrace)
}

func RaiseException() {
	const EXCEPTION_NONCONTINUABLE = 1
	mod := syscall.MustLoadDLL("kernel32.dll")
	proc := mod.MustFindProc("RaiseException")
	proc.Call(0xbad, EXCEPTION_NONCONTINUABLE, 0, 0)
	println("RaiseException should not return")
}

func ZeroDivisionException() {
	x := 1
	y := 0
	z := x / y
	println(z)
}

func getPagefileUsage() (uintptr, error) {
	p, err := syscall.GetCurrentProcess()
	if err != nil {
		return 0, err
	}
	var m windows.PROCESS_MEMORY_COUNTERS
	err = windows.GetProcessMemoryInfo(p, &m, uint32(unsafe.Sizeof(m)))
	if err != nil {
		return 0, err
	}
	return m.PagefileUsage, nil
}

func StackMemory() {
	mem1, err := getPagefileUsage()
	if err != nil {
		panic(err)
	}
	const threadCount = 100
	var wg sync.WaitGroup
	for i := 0; i < threadCount; i++ {
		wg.Add(1)
		go func() {
			runtime.LockOSThread()
			wg.Done()
			select {}
		}()
	}
	wg.Wait()
	mem2, err := getPagefileUsage()
	if err != nil {
		panic(err)
	}
	// assumes that this process creates 1 thread for each
	// thread locked goroutine plus extra 10 threads
	// like sysmon and others
	print((mem2 - mem1) / (threadCount + 10))
}

// CallbackInAnotherThreadTrace calls a Go callback from a series of
// threads created by Windows and writes an execution trace to stdout.
// After each call it sleeps long enough for sysmon to look at the P left
// behind by the callback. It sleeps instead of waiting for the thread in
// a system call, so that it cannot take that P back on syscall exit.
func CallbackInAnotherThreadTrace() {
	if runtime.GOMAXPROCS(0) < 2 {
		runtime.GOMAXPROCS(2)
	}
	createThread := syscall.MustLoadDLL("kernel32.dll").MustFindProc("CreateThread")
	cb := syscall.NewCallback(func(p uintptr) uintptr { return p })
	if err := trace.Start(os.Stdout); err != nil {
		panic(err)
	}
	for range 20 {
		r, _, err := createThread.Call(0, 0, cb, 0, 0, 0)
		if r == 0 {
			panic(err)
		}
		h := syscall.Handle(r)
		time.Sleep(30 * time.Millisecond)
		if _, err := syscall.WaitForSingleObject(h, syscall.INFINITE); err != nil {
			panic(err)
		}
		syscall.CloseHandle(h)
	}
	trace.Stop()
}
