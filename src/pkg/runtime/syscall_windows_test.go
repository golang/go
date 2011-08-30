// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"syscall"
	"unsafe"
	"testing"
)

func TestStdCall(t *testing.T) {
	type Rect struct {
		left, top, right, bottom int32
	}

	h, e := syscall.LoadLibrary("user32.dll")
	if e != 0 {
		t.Fatal("LoadLibrary(USER32)")
	}
	p, e := syscall.GetProcAddress(h, "UnionRect")
	if e != 0 {
		t.Fatal("GetProcAddress(USER32.UnionRect)")
	}

	res := Rect{}
	expected := Rect{1, 1, 40, 60}
	a, _, _ := syscall.Syscall(uintptr(p),
		3,
		uintptr(unsafe.Pointer(&res)),
		uintptr(unsafe.Pointer(&Rect{10, 1, 14, 60})),
		uintptr(unsafe.Pointer(&Rect{1, 2, 40, 50})))
	if a != 1 || res.left != expected.left ||
		res.top != expected.top ||
		res.right != expected.right ||
		res.bottom != expected.bottom {
		t.Error("stdcall USER32.UnionRect returns", a, "res=", res)
	}
}

func TestCDecl(t *testing.T) {
	h, e := syscall.LoadLibrary("user32.dll")
	if e != 0 {
		t.Fatal("LoadLibrary(USER32)")
	}
	p, e := syscall.GetProcAddress(h, "wsprintfA")
	if e != 0 {
		t.Fatal("GetProcAddress(USER32.wsprintfA)")
	}

	var buf [50]byte
	a, _, _ := syscall.Syscall6(uintptr(p),
		5,
		uintptr(unsafe.Pointer(&buf[0])),
		uintptr(unsafe.Pointer(syscall.StringBytePtr("%d %d %d"))),
		1000, 2000, 3000, 0)
	if string(buf[:a]) != "1000 2000 3000" {
		t.Error("cdecl USER32.wsprintfA returns", a, "buf=", buf[:a])
	}
}

func TestCallback(t *testing.T) {
	h, e := syscall.LoadLibrary("user32.dll")
	if e != 0 {
		t.Fatal("LoadLibrary(USER32)")
	}
	pEnumWindows, e := syscall.GetProcAddress(h, "EnumWindows")
	if e != 0 {
		t.Fatal("GetProcAddress(USER32.EnumWindows)")
	}
	pIsWindow, e := syscall.GetProcAddress(h, "IsWindow")
	if e != 0 {
		t.Fatal("GetProcAddress(USER32.IsWindow)")
	}
	counter := 0
	cb := syscall.NewCallback(func(hwnd syscall.Handle, lparam uintptr) uintptr {
		if lparam != 888 {
			t.Error("lparam was not passed to callback")
		}
		b, _, _ := syscall.Syscall(uintptr(pIsWindow), 1, uintptr(hwnd), 0, 0)
		if b == 0 {
			t.Error("USER32.IsWindow returns FALSE")
		}
		counter++
		return 1 // continue enumeration
	})
	a, _, _ := syscall.Syscall(uintptr(pEnumWindows), 2, cb, 888, 0)
	if a == 0 {
		t.Error("USER32.EnumWindows returns FALSE")
	}
	if counter == 0 {
		t.Error("Callback has been never called or your have no windows")
	}
}

func TestCallbackInAnotherThread(t *testing.T) {
	// TODO: test a function which calls back in another thread: QueueUserAPC() or CreateThread()
}
