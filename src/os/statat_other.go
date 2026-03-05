// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (js && wasm) || plan9

package os

func (f *File) lstatatNolog(name string) (FileInfo, error) {
	// These platforms don't have fstatat, so use stat instead.
	return Lstat(f.name + "/" + name)
}
