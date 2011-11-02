// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"syscall"
	"testing"
	"unsafe"
)

type DLL struct {
	*syscall.DLL
	t *testing.T
}

func GetDLL(t *testing.T, name string) *DLL {
	d, e := syscall.LoadDLL(name)
	if e != nil {
		t.Fatal(e)
	}
	return &DLL{DLL: d, t: t}
}

func (d *DLL) Proc(name string) *syscall.Proc {
	p, e := d.FindProc(name)
	if e != nil {
		d.t.Fatal(e)
	}
	return p
}

func TestStdCall(t *testing.T) {
	type Rect struct {
		left, top, right, bottom int32
	}
	res := Rect{}
	expected := Rect{1, 1, 40, 60}
	a, _, _ := GetDLL(t, "user32.dll").Proc("UnionRect").Call(
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

	d := GetDLL(t, "kernel32.dll")

	var m1, m2 uintptr
	VerSetConditionMask := d.Proc("VerSetConditionMask")
	m1, m2, _ = VerSetConditionMask.Call(m1, m2, VER_MAJORVERSION, VER_GREATER_EQUAL)
	m1, m2, _ = VerSetConditionMask.Call(m1, m2, VER_MINORVERSION, VER_GREATER_EQUAL)
	m1, m2, _ = VerSetConditionMask.Call(m1, m2, VER_SERVICEPACKMAJOR, VER_GREATER_EQUAL)
	m1, m2, _ = VerSetConditionMask.Call(m1, m2, VER_SERVICEPACKMINOR, VER_GREATER_EQUAL)

	vi := OSVersionInfoEx{
		MajorVersion:     5,
		MinorVersion:     1,
		ServicePackMajor: 2,
		ServicePackMinor: 0,
	}
	vi.OSVersionInfoSize = uint32(unsafe.Sizeof(vi))
	r, _, e2 := d.Proc("VerifyVersionInfoW").Call(
		uintptr(unsafe.Pointer(&vi)),
		VER_MAJORVERSION|VER_MINORVERSION|VER_SERVICEPACKMAJOR|VER_SERVICEPACKMINOR,
		m1, m2)
	if r == 0 && e2 != ERROR_OLD_WIN_VERSION {
		t.Errorf("VerifyVersionInfo failed: (%d) %s", e2, syscall.Errstr(int(e2)))
	}
}

func TestCDecl(t *testing.T) {
	var buf [50]byte
	a, _, _ := GetDLL(t, "user32.dll").Proc("wsprintfA").Call(
		uintptr(unsafe.Pointer(&buf[0])),
		uintptr(unsafe.Pointer(syscall.StringBytePtr("%d %d %d"))),
		1000, 2000, 3000)
	if string(buf[:a]) != "1000 2000 3000" {
		t.Error("cdecl USER32.wsprintfA returns", a, "buf=", buf[:a])
	}
}

func TestCallback(t *testing.T) {
	d := GetDLL(t, "user32.dll")
	isWindows := d.Proc("IsWindow")
	counter := 0
	cb := syscall.NewCallback(func(hwnd syscall.Handle, lparam uintptr) uintptr {
		if lparam != 888 {
			t.Error("lparam was not passed to callback")
		}
		b, _, _ := isWindows.Call(uintptr(hwnd))
		if b == 0 {
			t.Error("USER32.IsWindow returns FALSE")
		}
		counter++
		return 1 // continue enumeration
	})
	a, _, _ := d.Proc("EnumWindows").Call(cb, 888)
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
