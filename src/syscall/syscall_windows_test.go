// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall_test

import (
	"os"
	"path/filepath"
	"runtime"
	"syscall"
	"testing"
	"time"
	"unsafe"
)

func TestWin32finddata(t *testing.T) {
	dir, err := os.MkdirTemp("", "go-build")
	if err != nil {
		t.Fatalf("failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(dir)

	path := filepath.Join(dir, "long_name.and_extension")
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("failed to create %v: %v", path, err)
	}
	f.Close()

	type X struct {
		fd  syscall.Win32finddata
		got byte
		pad [10]byte // to protect ourselves

	}
	var want byte = 2 // it is unlikely to have this character in the filename
	x := X{got: want}

	pathp, _ := syscall.UTF16PtrFromString(path)
	h, err := syscall.FindFirstFile(pathp, &(x.fd))
	if err != nil {
		t.Fatalf("FindFirstFile failed: %v", err)
	}
	err = syscall.FindClose(h)
	if err != nil {
		t.Fatalf("FindClose failed: %v", err)
	}

	if x.got != want {
		t.Fatalf("memory corruption: want=%d got=%d", want, x.got)
	}
}

func abort(funcname string, err error) {
	panic(funcname + " failed: " + err.Error())
}

func ExampleLoadLibrary() {
	h, err := syscall.LoadLibrary("kernel32.dll")
	if err != nil {
		abort("LoadLibrary", err)
	}
	defer syscall.FreeLibrary(h)
	proc, err := syscall.GetProcAddress(h, "GetVersion")
	if err != nil {
		abort("GetProcAddress", err)
	}
	r, _, _ := syscall.Syscall(uintptr(proc), 0, 0, 0, 0)
	major := byte(r)
	minor := uint8(r >> 8)
	build := uint16(r >> 16)
	print("windows version ", major, ".", minor, " (Build ", build, ")\n")
}

func TestTOKEN_ALL_ACCESS(t *testing.T) {
	if syscall.TOKEN_ALL_ACCESS != 0xF01FF {
		t.Errorf("TOKEN_ALL_ACCESS = %x, want 0xF01FF", syscall.TOKEN_ALL_ACCESS)
	}
}

func TestProcThreadAttributeListPointers(t *testing.T) {
	list, err := syscall.NewProcThreadAttributeList(1)
	if err != nil {
		t.Errorf("unable to create ProcThreadAttributeList: %v", err)
	}
	done := make(chan struct{})
	fds := make([]syscall.Handle, 20)
	runtime.SetFinalizer(&fds[0], func(*syscall.Handle) {
		close(done)
	})
	err = syscall.UpdateProcThreadAttribute(list, 0, syscall.PROC_THREAD_ATTRIBUTE_HANDLE_LIST, unsafe.Pointer(&fds[0]), uintptr(len(fds))*unsafe.Sizeof(fds[0]), nil, nil)
	if err != nil {
		syscall.DeleteProcThreadAttributeList(list)
		t.Errorf("unable to update ProcThreadAttributeList: %v", err)
		return
	}
	runtime.GC()
	runtime.GC()
	select {
	case <-done:
		t.Error("ProcThreadAttributeList was garbage collected unexpectedly")
	default:
	}
	syscall.DeleteProcThreadAttributeList(list)
	runtime.GC()
	runtime.GC()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Error("ProcThreadAttributeList was not garbage collected after a second")
	}
}
