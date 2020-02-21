// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin,arm64,!go1.12

package unix

func Getdirentries(fd int, buf []byte, basep *uintptr) (n int, err error) {
	return 0, ENOSYS
}
