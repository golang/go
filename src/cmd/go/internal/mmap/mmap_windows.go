// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mmap

import (
	"fmt"
	"os"
	"syscall"
	"unsafe"

	"internal/syscall/windows"
)

func mmapFile(f *os.File) (Data, error) {
	st, err := f.Stat()
	if err != nil {
		return Data{}, err
	}
	size := st.Size()
	if size == 0 {
		return Data{f, nil}, nil
	}
	h, err := syscall.CreateFileMapping(syscall.Handle(f.Fd()), nil, syscall.PAGE_READONLY, 0, 0, nil)
	if err != nil {
		return Data{}, fmt.Errorf("CreateFileMapping %s: %w", f.Name(), err)
	}

	addr, err := syscall.MapViewOfFile(h, syscall.FILE_MAP_READ, 0, 0, 0)
	if err != nil {
		return Data{}, fmt.Errorf("MapViewOfFile %s: %w", f.Name(), err)
	}
	var info windows.MemoryBasicInformation
	err = windows.VirtualQuery(addr, &info, unsafe.Sizeof(info))
	if err != nil {
		return Data{}, fmt.Errorf("VirtualQuery %s: %w", f.Name(), err)
	}
	data := unsafe.Slice((*byte)(unsafe.Pointer(addr)), int(info.RegionSize))
	return Data{f, data}, nil
}
