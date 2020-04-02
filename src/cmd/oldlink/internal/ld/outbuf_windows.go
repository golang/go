// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"reflect"
	"syscall"
	"unsafe"
)

func (out *OutBuf) Mmap(filesize uint64) error {
	err := out.f.Truncate(int64(filesize))
	if err != nil {
		Exitf("resize output file failed: %v", err)
	}

	low, high := uint32(filesize), uint32(filesize>>32)
	fmap, err := syscall.CreateFileMapping(syscall.Handle(out.f.Fd()), nil, syscall.PAGE_READONLY, high, low, nil)
	if err != nil {
		return err
	}
	defer syscall.CloseHandle(fmap)

	ptr, err := syscall.MapViewOfFile(fmap, syscall.FILE_MAP_READ|syscall.FILE_MAP_WRITE, 0, 0, uintptr(filesize))
	if err != nil {
		return err
	}
	*(*reflect.SliceHeader)(unsafe.Pointer(&out.buf)) = reflect.SliceHeader{Data: ptr, Len: int(filesize), Cap: int(filesize)}
	return nil
}

func (out *OutBuf) Munmap() {
	if out.buf == nil {
		return
	}
	err := syscall.UnmapViewOfFile(uintptr(unsafe.Pointer(&out.buf[0])))
	if err != nil {
		Exitf("UnmapViewOfFile failed: %v", err)
	}
}

func (out *OutBuf) Msync() error {
	if out.buf == nil {
		return nil
	}
	return syscall.FlushViewOfFile(uintptr(unsafe.Pointer(&out.buf[0])), 0)
}
