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
// The boolean result indicates whether the file was opened.
// If it is true, the caller should avoid attempting
// to write to the file on Windows, because Windows locks
// the open file, and writes to it will fail.
func Mmap(file string) (Data, bool, error) {
	f, err := os.Open(file)
	if err != nil {
		return Data{}, false, err
	}
	data, err := mmapFile(f)

	// Closing the file causes it not to count against this process's
	// limit on open files; however, the mapping still counts against
	// the system-wide limit, which is typically higher. Examples:
	//
	//     macOS process (sysctl kern.maxfilesperproc):  61440
	//     macOS system  (sysctl kern.maxfiles):        122880
	//     linux process (ulimit -n)                   1048576
	//     linux system  (/proc/sys/fs/file-max)        100000
	if cerr := f.Close(); cerr != nil && err == nil {
		return data, true, cerr
	}

	// The file is still considered to be in use on Windows after
	// it's closed because of the mapping.
	return data, true, err
}
