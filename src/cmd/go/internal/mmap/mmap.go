// Copyright 2011 The Go Authors. All rights reserved.
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

// Data is mmap'ed read-only data from a file.
// The backing file is never closed, so Data
// remains valid for the lifetime of the process.
type Data struct {
	f    *os.File
	Data []byte
}

// Mmap maps the given file into memory.
func Mmap(file string) (Data, bool, error) {
	f, err := os.Open(file)
	if err != nil {
		return Data{}, false, err
	}
	data, err := mmapFile(f)
	return data, true, err
}
