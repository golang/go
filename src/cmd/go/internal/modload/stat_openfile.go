// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (js && wasm) || plan9
// +build js,wasm plan9

// On plan9, per http://9p.io/magic/man2html/2/access: “Since file permissions
// are checked by the server and group information is not known to the client,
// access must open the file to check permissions.”
//
// aix and js,wasm are similar, in that they do not define syscall.Access.

package modload

import (
	"io/fs"
	"os"
)

// hasWritePerm reports whether the current user has permission to write to the
// file with the given info.
func hasWritePerm(path string, _ fs.FileInfo) bool {
	if f, err := os.OpenFile(path, os.O_WRONLY, 0); err == nil {
		f.Close()
		return true
	}
	return false
}
