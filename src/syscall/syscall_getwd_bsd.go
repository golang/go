// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build dragonfly freebsd netbsd openbsd

package syscall

const ImplementsGetwd = true

func Getwd() (string, error) {
	var buf [pathMax]byte
	_, err := getcwd(buf[:])
	if err != nil {
		return "", err
	}
	n := clen(buf[:])
	if n < 1 {
		return "", EINVAL
	}
	return string(buf[:n]), nil
}
