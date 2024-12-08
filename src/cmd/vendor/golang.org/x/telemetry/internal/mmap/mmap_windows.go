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

func mmapFile(f *os.File) (*Data, error) {
	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	size := st.Size()
	if size == 0 {
		return &Data{f, nil, nil}, nil
	}
	// set the min and max sizes to zero to map the whole file, as described in
	// https://learn.microsoft.com/en-us/windows/win32/memory/creating-a-file-mapping-object#file-mapping-size
	h, err := windows.CreateFileMapping(windows.Handle(f.Fd()), nil, syscall.PAGE_READWRITE, 0, 0, nil)
	if err != nil {
		return nil, fmt.Errorf("CreateFileMapping %s: %w", f.Name(), err)
	}
	// the mapping extends from zero to the end of the file mapping
	// https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-mapviewoffile
	addr, err := windows.MapViewOfFile(h, syscall.FILE_MAP_READ|syscall.FILE_MAP_WRITE, 0, 0, 0)
	if err != nil {
		return nil, fmt.Errorf("MapViewOfFile %s: %w", f.Name(), err)
	}
	// Note: previously, we called windows.VirtualQuery here to get the exact
	// size of the memory mapped region, but VirtualQuery reported sizes smaller
	// than the actual file size (hypothesis: VirtualQuery only reports pages in
	// a certain state, and newly written pages may not be counted).
	return &Data{f, unsafe.Slice((*byte)(unsafe.Pointer(addr)), size), h}, nil
}

func munmapFile(d *Data) error {
	err := windows.UnmapViewOfFile(uintptr(unsafe.Pointer(&d.Data[0])))
	x, ok := d.Windows.(windows.Handle)
	if ok {
		windows.CloseHandle(x)
	}
	d.f.Close()
	return err
}
