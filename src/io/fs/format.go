// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fs

import (
	"time"
)

// FormatFileInfo returns a formatted version of info for human readability.
// Implementations of [FileInfo] can call this from a String method.
// The output for a file named "hello.go", 100 bytes, mode 0o644, created
// January 1, 1970 at noon is
//
//	-rw-r--r-- 100 1970-01-01 12:00:00 hello.go
func FormatFileInfo(info FileInfo) string {
	name := info.Name()
	b := make([]byte, 0, 40+len(name))
	b = append(b, info.Mode().String()...)
	b = append(b, ' ')

	size := info.Size()
	var usize uint64
	if size >= 0 {
		usize = uint64(size)
	} else {
		b = append(b, '-')
		usize = uint64(-size)
	}
	var buf [20]byte
	i := len(buf) - 1
	for usize >= 10 {
		q := usize / 10
		buf[i] = byte('0' + usize - q*10)
		i--
		usize = q
	}
	buf[i] = byte('0' + usize)
	b = append(b, buf[i:]...)
	b = append(b, ' ')

	b = append(b, info.ModTime().Format(time.DateTime)...)
	b = append(b, ' ')

	b = append(b, name...)
	if info.IsDir() {
		b = append(b, '/')
	}

	return string(b)
}

// FormatDirEntry returns a formatted version of dir for human readability.
// Implementations of [DirEntry] can call this from a String method.
// The outputs for a directory named subdir and a file named hello.go are:
//
//	d subdir/
//	- hello.go
func FormatDirEntry(dir DirEntry) string {
	name := dir.Name()
	b := make([]byte, 0, 5+len(name))

	// The Type method does not return any permission bits,
	// so strip them from the string.
	mode := dir.Type().String()
	mode = mode[:len(mode)-9]

	b = append(b, mode...)
	b = append(b, ' ')
	b = append(b, name...)
	if dir.IsDir() {
		b = append(b, '/')
	}
	return string(b)
}
