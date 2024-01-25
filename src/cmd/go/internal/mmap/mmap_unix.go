// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package mmap

import (
	"fmt"
	"io/fs"
	"os"
	"syscall"
)

func mmapFile(f *os.File) (Data, error) {
	st, err := f.Stat()
	if err != nil {
		return Data{}, err
	}
	size := st.Size()
	pagesize := int64(os.Getpagesize())
	if int64(int(size+(pagesize-1))) != size+(pagesize-1) {
		return Data{}, fmt.Errorf("%s: too large for mmap", f.Name())
	}
	n := int(size)
	if n == 0 {
		return Data{f, nil}, nil
	}
	mmapLength := int(((size + pagesize - 1) / pagesize) * pagesize) // round up to page size
	data, err := syscall.Mmap(int(f.Fd()), 0, mmapLength, syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return Data{}, &fs.PathError{Op: "mmap", Path: f.Name(), Err: err}
	}
	return Data{f, data[:n]}, nil
}
