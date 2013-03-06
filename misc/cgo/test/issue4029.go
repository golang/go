// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !windows

package cgotest

/*
#include <dlfcn.h>
#cgo linux LDFLAGS: -ldl
*/
import "C"

import (
	"fmt"
	"testing"
)

//export IMPIsOpaque
func IMPIsOpaque() {
	fmt.Println("isOpaque")
}

//export IMPInitWithFrame
func IMPInitWithFrame() {
	fmt.Println("IInitWithFrame")
}

//export IMPDrawRect
func IMPDrawRect() {
	fmt.Println("drawRect:")
}

//export IMPWindowResize
func IMPWindowResize() {
	fmt.Println("windowDidResize:")
}

func test4029(t *testing.T) {
	loadThySelf(t, "IMPWindowResize")
	loadThySelf(t, "IMPDrawRect")
	loadThySelf(t, "IMPInitWithFrame")
	loadThySelf(t, "IMPIsOpaque")
}

func loadThySelf(t *testing.T, symbol string) {
	this_process := C.dlopen(nil, C.RTLD_NOW)
	if this_process == nil {
		t.Error("dlopen:", C.GoString(C.dlerror()))
		return
	}
	defer C.dlclose(this_process)

	symbol_address := C.dlsym(this_process, C.CString(symbol))
	if symbol_address == nil {
		t.Error("dlsym:", C.GoString(C.dlerror()))
		return
	}
	t.Log(symbol, symbol_address)
}
