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

func Test64BitReturnStdCall(t *testing.T) {

	const (
		VER_BUILDNUMBER      = 0x0000004
		VER_MAJORVERSION     = 0x0000002
		VER_MINORVERSION     = 0x0000001
		VER_PLATFORMID       = 0x0000008
		VER_PRODUCT_TYPE     = 0x0000080
		VER_SERVICEPACKMAJOR = 0x0000020
		VER_SERVICEPACKMINOR = 0x0000010
		VER_SUITENAME        = 0x0000040

		VER_EQUAL         = 1
		VER_GREATER       = 2
		VER_GREATER_EQUAL = 3
		VER_LESS          = 4
		VER_LESS_EQUAL    = 5

		ERROR_OLD_WIN_VERSION = 1150
	)

	type OSVersionInfoEx struct {
		OSVersionInfoSize uint32
		MajorVersion      uint32
		MinorVersion      uint32
		BuildNumber       uint32
		PlatformId        uint32
		CSDVersion        [128]uint16
		ServicePackMajor  uint16
		ServicePackMinor  uint16
		SuiteMask         uint16
		ProductType       byte
		Reserve           byte
	}

	kernel32, e := syscall.LoadLibrary("kernel32.dll")
	if e != 0 {
		t.Fatalf("LoadLibrary(kernel32.dll) failed: %s", syscall.Errstr(e))
	}
	setMask, e := syscall.GetProcAddress(kernel32, "VerSetConditionMask")
	if e != 0 {
		t.Fatalf("GetProcAddress(kernel32.dll, VerSetConditionMask) failed: %s", syscall.Errstr(e))
	}
	verifyVersion, e := syscall.GetProcAddress(kernel32, "VerifyVersionInfoW")
	if e != 0 {
		t.Fatalf("GetProcAddress(kernel32.dll, VerifyVersionInfoW) failed: %s", syscall.Errstr(e))
	}

	var m1, m2 uintptr
	m1, m2, _ = syscall.Syscall6(setMask, 4, m1, m2, VER_MAJORVERSION, VER_GREATER_EQUAL, 0, 0)
	m1, m2, _ = syscall.Syscall6(setMask, 4, m1, m2, VER_MINORVERSION, VER_GREATER_EQUAL, 0, 0)
	m1, m2, _ = syscall.Syscall6(setMask, 4, m1, m2, VER_SERVICEPACKMAJOR, VER_GREATER_EQUAL, 0, 0)
	m1, m2, _ = syscall.Syscall6(setMask, 4, m1, m2, VER_SERVICEPACKMINOR, VER_GREATER_EQUAL, 0, 0)

	vi := OSVersionInfoEx{
		MajorVersion:     5,
		MinorVersion:     1,
		ServicePackMajor: 2,
		ServicePackMinor: 0,
	}
	vi.OSVersionInfoSize = uint32(unsafe.Sizeof(vi))
	r, _, e2 := syscall.Syscall6(verifyVersion,
		4,
		uintptr(unsafe.Pointer(&vi)),
		VER_MAJORVERSION|VER_MINORVERSION|VER_SERVICEPACKMAJOR|VER_SERVICEPACKMINOR,
		m1, m2, 0, 0)
	if r == 0 && e2 != ERROR_OLD_WIN_VERSION {
		t.Errorf("VerifyVersionInfo failed: (%d) %s", e2, syscall.Errstr(int(e2)))
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
