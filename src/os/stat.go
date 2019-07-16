// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import "internal/testlog"

// Stat returns a FileInfo describing the named file.
// If there is an error, it will be of type *PathError.
func Stat(name string) (FileInfo, error) {
	testlog.Stat(name)
	return statNolog(name)
}

// Lstat returns a FileInfo describing the named file.
// If the file is a symbolic link, the returned FileInfo
// describes the symbolic link. Lstat makes no attempt to follow the link.
// If there is an error, it will be of type *PathError.
func Lstat(name string) (FileInfo, error) {
	testlog.Stat(name)
	return lstatNolog(name)
}
