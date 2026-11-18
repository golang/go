// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

/*
#include "cgoopt.h"

#cgo nocallback opt_scalar
#cgo nocallback opt_aggregate
#cgo nocallback opt_errno
#cgo nocallback opt_retain
#cgo nocallback opt_retained
#cgo nocallback opt_forget
#cgo nocallback opt_wait
#cgo nocallback opt_entered
#cgo nocallback opt_release
*/
import "C"

import (
	"fmt"
	"runtime"
	"syscall"
	"unsafe"
)

//go:noinline
func grow(depth int) byte {
	var frame [4096]byte
	frame[depth] = byte(depth)
	var child byte
	if depth > 0 {
		child = grow(depth - 1)
	}
	runtime.KeepAlive(&frame)
	return frame[depth] + child
}

// stackAddress observes the marker without making it escape. ASan otherwise
// heap-allocates it at the unsafe.Pointer conversion, hiding stack movement.
//
//go:noinline
//go:nocheckptr
func stackAddress(p *byte) uintptr {
	return uintptr(unsafe.Pointer(p))
}

//export CgoOptGrow
func CgoOptGrow(value C.int) C.int {
	var marker byte
	before := stackAddress(&marker)
	if got := grow(64); got != 32 {
		panic(fmt.Sprintf("callback recursion = %d, want 32", got))
	}
	if after := stackAddress(&marker); after == before {
		panic("callback did not move the Go stack")
	}
	return value + 1
}

func checkCallback() {
	done := make(chan C.int, 1)
	go func() { done <- C.opt_callback(40) }()
	if got := <-done; got != 42 {
		panic(fmt.Sprintf("callback result = %d, want 42", got))
	}
}

func checkResults() {
	stop := make(chan struct{})
	stopped := make(chan struct{})
	go func() {
		defer close(stopped)
		for {
			select {
			case <-stop:
				return
			default:
				runtime.GC()
			}
		}
	}()
	defer func() {
		close(stop)
		<-stopped
	}()
	for i := range 2000 {
		if got := C.opt_scalar(C.uint64_t(i)); got != C.uint64_t(i+1) {
			panic(fmt.Sprintf("scalar result = %d at %d", got, i))
		}
		pair := C.opt_aggregate(C.uint64_t(i))
		if pair.whole != C.uint64_t(i) || pair.fraction != C.double(float64(i)+0.5) {
			panic(fmt.Sprintf("aggregate result = %v at %d", pair, i))
		}
		value := C.int(i)
		if i%2 == 0 {
			value = -1
		}
		got, err := C.opt_errno(value)
		if got != value || value < 0 && err != syscall.Errno(C.EINVAL) || value >= 0 && err != nil {
			panic(fmt.Sprintf("errno result = %d, %v for %d", got, err, value))
		}
	}
}

func checkRetainedPointer() {
	value := new(C.int)
	*value = 73
	var pin runtime.Pinner
	pin.Pin(value)
	defer pin.Unpin()
	if got := C.opt_retain(value); got != value {
		panic("retained pointer result changed address")
	}
	defer C.opt_forget()
	value = nil
	runtime.GC()
	if got := C.opt_retained(); got != 73 {
		panic(fmt.Sprintf("retained pointer value = %d, want 73", got))
	}
}

func checkProgress() {
	previous := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(previous)
	done := make(chan struct{})
	go func() {
		for C.opt_entered() == 0 {
			runtime.Gosched()
		}
		// Complete a collection while the other goroutine's result frame is
		// suspended in the callback-free C call.
		runtime.GC()
		C.opt_release()
		close(done)
	}()
	if got := C.opt_wait(); got != 42 {
		panic("blocked nocallback call prevented scheduler progress")
	}
	<-done
}

func main() {
	previous := runtime.GOMAXPROCS(2)
	defer runtime.GOMAXPROCS(previous)
	checkCallback()
	checkResults()
	checkRetainedPointer()
	checkProgress()
	fmt.Println("callback stack moved; results, errno, retained pointer, and scheduler progress ok")
}
