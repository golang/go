// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !windows,!static

package cgotest

/*
#include <stdint.h>
#include <dlfcn.h>
#cgo linux LDFLAGS: -ldl

extern uintptr_t dlopen4029(char*, int);
extern uintptr_t dlsym4029(uintptr_t, char*);
extern int dlclose4029(uintptr_t);

extern void call4029(uintptr_t arg);
*/
import "C"

import (
	"testing"
)

var callbacks int

//export IMPIsOpaque
func IMPIsOpaque() {
	callbacks++
}

//export IMPInitWithFrame
func IMPInitWithFrame() {
	callbacks++
}

//export IMPDrawRect
func IMPDrawRect() {
	callbacks++
}

//export IMPWindowResize
func IMPWindowResize() {
	callbacks++
}

func test4029(t *testing.T) {
	loadThySelf(t, "IMPWindowResize")
	loadThySelf(t, "IMPDrawRect")
	loadThySelf(t, "IMPInitWithFrame")
	loadThySelf(t, "IMPIsOpaque")
	if callbacks != 4 {
		t.Errorf("got %d callbacks, expected 4", callbacks)
	}
}

func loadThySelf(t *testing.T, symbol string) {
	this_process := C.dlopen4029(nil, C.RTLD_NOW)
	if this_process == 0 {
		t.Error("dlopen:", C.GoString(C.dlerror()))
		return
	}
	defer C.dlclose4029(this_process)

	symbol_address := C.dlsym4029(this_process, C.CString(symbol))
	if symbol_address == 0 {
		t.Error("dlsym:", C.GoString(C.dlerror()))
		return
	}
	t.Log(symbol, symbol_address)
	C.call4029(symbol_address)
}
