// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This package is a lightly modified version of the mmap code
// in github.com/google/codesearch/index.

// The mmap package provides an abstraction for memory mapping files
// on different platforms.
package mmap

import (
	"os"
)

// The backing file is never closed, so Data
// remains valid for the lifetime of the process.
type Data struct {
	// TODO(pjw): might be better to define versions of Data
	// for the 3 specializations
	f    *os.File
	Data []byte
	// Some windows magic
	Windows interface{}
}

// Mmap maps the given file into memory.
// When remapping a file, pass the most recently returned Data.
func Mmap(f *os.File) (*Data, error) {
	return mmapFile(f)
}

// Munmap unmaps the given file from memory.
func Munmap(d *Data) error {
	return munmapFile(d)
}
