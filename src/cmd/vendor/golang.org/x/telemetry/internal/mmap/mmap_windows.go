// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mmap

import (
	"fmt"
	"os"
	"syscall"
	"unsafe"

	"golang.org/x/sys/windows"
)

func mmapFile(f *os.File, previous *Data) (Data, error) {
	if previous != nil {
		munmapFile(*previous)
	}
	st, err := f.Stat()
	if err != nil {
		return Data{}, err
	}
	size := st.Size()
	if size == 0 {
		return Data{f, nil, nil}, nil
	}
	h, err := windows.CreateFileMapping(windows.Handle(f.Fd()), nil, syscall.PAGE_READWRITE, 0, 0, nil)
	if err != nil {
		return Data{}, fmt.Errorf("CreateFileMapping %s: %w", f.Name(), err)
	}

	addr, err := windows.MapViewOfFile(h, syscall.FILE_MAP_READ|syscall.FILE_MAP_WRITE, 0, 0, 0)
	if err != nil {
		return Data{}, fmt.Errorf("MapViewOfFile %s: %w", f.Name(), err)
	}
	var info windows.MemoryBasicInformation
	err = windows.VirtualQuery(addr, &info, unsafe.Sizeof(info))
	if err != nil {
		return Data{}, fmt.Errorf("VirtualQuery %s: %w", f.Name(), err)
	}
	data := unsafe.Slice((*byte)(unsafe.Pointer(addr)), int(info.RegionSize))
	return Data{f, data, h}, nil
}

func munmapFile(d Data) error {
	err := windows.UnmapViewOfFile(uintptr(unsafe.Pointer(&d.Data[0])))
	x, ok := d.Windows.(windows.Handle)
	if ok {
		windows.CloseHandle(x)
	}
	d.f.Close()
	return err
}
