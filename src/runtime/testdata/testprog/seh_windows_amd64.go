// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"internal/syscall/windows"
	"runtime"
	"syscall"
	"time"
	"unsafe"
)

func init() {
	register("SehUnwindParkedThread", SehUnwindParkedThread)
}

// SehUnwindParkedThread walks the stack of a thread parked in the
// scheduler with the Windows native unwinder, as debuggers, profilers
// and injected DLLs do. The thread is parked via mcall, so the walk
// goes through mcall's unwind info on g0. See go.dev/issue/81885.
func SehUnwindParkedThread() {
	kernel32 := syscall.MustLoadDLL("kernel32.dll")
	getCurrentThreadId := kernel32.MustFindProc("GetCurrentThreadId")
	openThread := kernel32.MustFindProc("OpenThread")
	suspendThread := kernel32.MustFindProc("SuspendThread")
	resumeThread := kernel32.MustFindProc("ResumeThread")
	getThreadContext := kernel32.MustFindProc("GetThreadContext")

	tid := make(chan uintptr)
	block := make(chan struct{})
	defer close(block)
	go func() {
		// Blocking on a channel while locked to the thread parks
		// the thread: gopark -> mcall(park_m) -> stoplockedm.
		runtime.LockOSThread()
		id, _, _ := getCurrentThreadId.Call()
		tid <- id
		<-block
	}()
	id := <-tid

	const (
		THREAD_SUSPEND_RESUME = 0x0002
		THREAD_GET_CONTEXT    = 0x0008
		CONTEXT_FULL          = 0x10000b
	)
	h, _, err := openThread.Call(THREAD_SUSPEND_RESUME|THREAD_GET_CONTEXT, 0, id)
	if h == 0 {
		panic(err)
	}
	defer syscall.CloseHandle(syscall.Handle(h))

	// GetThreadContext requires a 16-byte aligned CONTEXT.
	buf := make([]byte, unsafe.Sizeof(windows.Context{})+16)
	ctx := (*windows.Context)(unsafe.Pointer((uintptr(unsafe.Pointer(&buf[0])) + 15) &^ 15))

	// The goroutine may not have parked yet, so retry until the walk
	// goes through mcall.
	for try := 0; try < 100; try++ {
		if r, _, err := suspendThread.Call(h); r == ^uintptr(0) {
			panic(err)
		}
		ctx.ContextFlags = CONTEXT_FULL
		if r, _, err := getThreadContext.Call(h, uintptr(unsafe.Pointer(ctx))); r == 0 {
			panic(err)
		}
		sawMcall := false
		for i := 0; i < 64 && ctx.PC() != 0; i++ {
			pc := ctx.PC()
			if f := runtime.FuncForPC(pc - 1); f != nil && f.Name() == "runtime.mcall" {
				sawMcall = true
			}
			var base, frame uintptr
			fn := windows.RtlLookupFunctionEntry(pc, &base, nil)
			if fn == nil {
				// Leaf function: the return address is at SP.
				ctx.SetPC(*(*uintptr)(unsafe.Pointer(ctx.SP())))
				ctx.SetSP(ctx.SP() + 8)
				continue
			}
			windows.RtlVirtualUnwind(0, base, pc, fn, unsafe.Pointer(ctx), nil, &frame, nil)
		}
		resumeThread.Call(h)
		if sawMcall {
			println("OK")
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	println("thread never parked in mcall")
}
