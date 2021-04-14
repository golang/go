// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"reflect"
	"syscall"
	"unsafe"
)

// Mmap maps the output file with the given size. It unmaps the old mapping
// if it is already mapped. It also flushes any in-heap data to the new
// mapping.
func (out *OutBuf) Mmap(filesize uint64) error {
	oldlen := len(out.buf)
	if oldlen != 0 {
		out.munmap()
	}

	err := out.f.Truncate(int64(filesize))
	if err != nil {
		Exitf("resize output file failed: %v", err)
	}

	low, high := uint32(filesize), uint32(filesize>>32)
	fmap, err := syscall.CreateFileMapping(syscall.Handle(out.f.Fd()), nil, syscall.PAGE_READWRITE, high, low, nil)
	if err != nil {
		return err
	}
	defer syscall.CloseHandle(fmap)

	ptr, err := syscall.MapViewOfFile(fmap, syscall.FILE_MAP_READ|syscall.FILE_MAP_WRITE, 0, 0, uintptr(filesize))
	if err != nil {
		return err
	}
	bufHdr := (*reflect.SliceHeader)(unsafe.Pointer(&out.buf))
	bufHdr.Data = ptr
	bufHdr.Len = int(filesize)
	bufHdr.Cap = int(filesize)

	// copy heap to new mapping
	if uint64(oldlen+len(out.heap)) > filesize {
		panic("mmap size too small")
	}
	copy(out.buf[oldlen:], out.heap)
	out.heap = out.heap[:0]
	return nil
}

func (out *OutBuf) munmap() {
	if out.buf == nil {
		return
	}
	// Apparently unmapping without flush may cause ACCESS_DENIED error
	// (see issue 38440).
	err := syscall.FlushViewOfFile(uintptr(unsafe.Pointer(&out.buf[0])), 0)
	if err != nil {
		Exitf("FlushViewOfFile failed: %v", err)
	}
	err = syscall.UnmapViewOfFile(uintptr(unsafe.Pointer(&out.buf[0])))
	out.buf = nil
	if err != nil {
		Exitf("UnmapViewOfFile failed: %v", err)
	}
}
