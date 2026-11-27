// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jit && amd64

package runtime_test

import (
	"reflect"
	"runtime"
	"runtime/jit"
	"testing"
	"unsafe"
)

//go:noinline
func userFrameProfileCaller() {}

var userFrameProfileStack [3]uintptr

func TestUserFrameMorestackTraceback(t *testing.T) {
	code := new([64]byte)
	pc := uintptr(unsafe.Pointer(code))
	handle := jit.Register(jit.Region{Start: pc, End: pc + 64, Unwind: jit.UnwindDeclare})
	defer handle.Unregister()
	stackMap := jit.StackMap{FrameWords: 2, PointerMask: []byte{0}, HasUnwind: true, CallerPCOffset: 0, CallerSPOffset: unsafe.Sizeof(uintptr(0))}
	nextMap := stackMap
	nextMap.PCOffset = 16
	handle.AddStackMaps(stackMap, nextMap)
	caller := reflect.ValueOf(userFrameProfileCaller).Pointer()
	stack := &userFrameProfileStack
	sp := uintptr(unsafe.Pointer(&stack[0]))
	for _, test := range []struct {
		name                           string
		savedPC                        uintptr
		stack                          [3]uintptr
		resolved, scannedPC, scannedSP uintptr
	}{
		{"user_frame", pc, [3]uintptr{caller, 0, 0}, caller, pc, sp},
		{"user_frame_chain", pc, [3]uintptr{pc + 16, caller, 0}, caller, pc, sp},
		{"go_frame", caller, [3]uintptr{0, 0, 0}, caller, 0, 0},
		{"unknown_profile_pc", 1, [3]uintptr{0, 0, 0}, 0, 0, 0},
	} {
		t.Run(test.name, func(t *testing.T) {
			*stack = test.stack
			got, scannedPC, scannedSP := runtime.UserFrameMorestackForTest(test.savedPC, sp)
			if got != test.resolved || scannedPC != test.scannedPC || scannedSP != test.scannedSP {
				t.Fatalf("morestack transition = (%#x, %#x, %#x), want (%#x, %#x, %#x)", got, scannedPC, scannedSP, test.resolved, test.scannedPC, test.scannedSP)
			}
		})
	}
	runtime.KeepAlive(code)
	runtime.KeepAlive(stack)
}
