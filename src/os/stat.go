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
//
// On Windows, if the file is a reparse point that is a surrogate for another
// named entity (such as a symbolic link or mounted folder), the returned
// FileInfo describes the reparse point, and makes no attempt to resolve it.
func Lstat(name string) (FileInfo, error) {
	testlog.Stat(name)
	return lstatNolog(name)
}
