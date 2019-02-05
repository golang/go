// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "C"
import (
	"internal/syscall/windows"
	"runtime"
	"sync"
	"syscall"
	"unsafe"
)

func init() {
	register("StackMemory", StackMemory)
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
	print((mem2 - mem1) / threadCount)
}
