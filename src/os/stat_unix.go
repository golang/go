// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build aix darwin dragonfly freebsd js,wasm linux netbsd openbsd solaris

package os

import (
	"syscall"
)

// Stat returns the FileInfo structure describing file.
// If there is an error, it will be of type *PathError.
func (f *File) Stat() (FileInfo, error) {
	if f == nil {
		return nil, ErrInvalid
	}
	var fs fileStat
	err := f.pfd.Fstat(&fs.sys)
	if err != nil {
		return nil, &PathError{"stat", f.name, err}
	}
	fillFileStatFromSys(&fs, f.name)
	return &fs, nil
}

// statNolog stats a file with no test logging.
func statNolog(name string) (FileInfo, error) {
	var fs fileStat
	err := syscall.Stat(name, &fs.sys)
	if err != nil {
		return nil, &PathError{"stat", name, err}
	}
	fillFileStatFromSys(&fs, name)
	return &fs, nil
}

// lstatNolog lstats a file with no test logging.
func lstatNolog(name string) (FileInfo, error) {
	var fs fileStat
	err := syscall.Lstat(name, &fs.sys)
	if err != nil {
		return nil, &PathError{"lstat", name, err}
	}
	fillFileStatFromSys(&fs, name)
	return &fs, nil
}
