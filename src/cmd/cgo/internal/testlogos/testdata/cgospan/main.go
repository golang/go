// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

/*
#include <stdlib.h>

static void span_consume(void *p) { (void)p; }
extern void *span_export(void);
static void span_roundtrip(void) { (void)span_export(); }

#cgo noescape span_stack
#cgo nocallback span_stack
static void span_stack(void *p) { (void)p; }
*/
import "C"

import (
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"unsafe"
)

type object struct {
	data [64]byte
	next *[64]byte
}

var global object
var globalBytes [64]byte

func consume(p unsafe.Pointer) {
	C.span_consume(p)
}

func roundtrip() {
	C.span_roundtrip()
}

func rejects(call func()) (rejected bool) {
	defer func() {
		if value := recover(); value != nil {
			if !strings.Contains(fmt.Sprint(value), "unpinned Go") {
				panic(value)
			}
			rejected = true
		}
	}()
	call()
	return false
}

func mustReject(name string, call func()) {
	if !rejects(call) {
		panic(name + ": accepted an unpinned Go pointer")
	}
}

func checkMemoryKinds() {
	consume(nil)
	value := new([64]byte)
	consume(unsafe.Pointer(value))
	consume(unsafe.Pointer(&globalBytes[31]))
	runtime.KeepAlive(value)

	p := C.malloc(64)
	if p == nil {
		panic("C allocation failed")
	}
	defer C.free(p)
	consume(p)

	var stack [64]byte
	stackPointer := unsafe.Pointer(&stack[31])
	C.span_stack(stackPointer)
	runtime.KeepAlive(&stack)
}

func checkInteriorAndGlobal() {
	value := &object{next: new([64]byte)}
	p := unsafe.Pointer(&value.data[len(value.data)-1])
	mustReject("unknown interior pointer", func() { consume(p) })
	var pin runtime.Pinner
	pin.Pin(value.next)
	defer pin.Unpin()
	consume(p)
	runtime.KeepAlive(value)

	global.next = new([64]byte)
	defer func() { global.next = nil }()
	pin.Pin(global.next)
	gp := unsafe.Pointer(&global.data[31])
	mustReject("unknown global extent", func() { consume(gp) })
}

func checkExportedResult(root, child bool) {
	value := &object{next: new([64]byte)}
	var pin runtime.Pinner
	defer pin.Unpin()
	if root {
		pin.Pin(value)
	}
	if child {
		pin.Pin(value.next)
	}
	exported = unsafe.Pointer(value)
	defer func() { exported = nil }()
	roundtrip()
	runtime.KeepAlive(value)
}

func mustRejectExport(name string) {
	executable, err := os.Executable()
	if err != nil {
		panic(err)
	}
	output, err := exec.Command(executable, name).CombinedOutput()
	if err == nil {
		panic(name + ": accepted an unpinned Go pointer")
	}
	if _, ok := err.(*exec.ExitError); !ok {
		panic(fmt.Sprintf("%s: could not run child: %v", name, err))
	}
	text := string(output)
	resultRejected := strings.Contains(text, "result of Go function") &&
		strings.Contains(text, "called from cgo is unpinned Go") &&
		strings.Contains(text, "or points to unpinned Go")
	// cgocheck2 checks the pointer store into the C frame before the
	// result check. An unpinned root then causes a throw, not a panic.
	storeRejected := name == "export-unpinned-root" &&
		strings.Contains(text, "fatal error: unpinned Go pointer stored into non-Go memory")
	if !resultRejected && !storeRejected {
		panic(fmt.Sprintf("%s: missing cgo rejection: %v\n%s", name, err, output))
	}
}

func checkExportedResults() {
	mustRejectExport("export-unpinned-root")
	mustRejectExport("export-unpinned-child")
	checkExportedResult(true, true)
}

func checkConcurrentGC() {
	value := &object{next: new([64]byte)}
	bytes := new([64]byte)
	var pin runtime.Pinner
	pin.Pin(value.next)
	defer pin.Unpin()
	stop := make(chan struct{})
	done := make(chan struct{})
	go func() {
		defer close(done)
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
		<-done
	}()
	p := unsafe.Pointer(&value.data[31])
	for range 2000 {
		consume(p)
		consume(unsafe.Pointer(bytes))
	}
	runtime.KeepAlive(value)
	runtime.KeepAlive(bytes)
}

func main() {
	if len(os.Args) == 2 {
		switch os.Args[1] {
		case "export-unpinned-root":
			checkExportedResult(false, true)
		case "export-unpinned-child":
			checkExportedResult(true, false)
		default:
			panic("unknown exported-result case")
		}
		return
	}
	if len(os.Args) != 1 {
		panic("unexpected arguments")
	}
	checkMemoryKinds()
	checkInteriorAndGlobal()
	checkExportedResults()
	checkConcurrentGC()
	fmt.Println("span pointer and exported-result checks ok")
}
