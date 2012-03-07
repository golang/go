// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package filepath

import (
	"syscall"
)

func evalSymlinks(path string) (string, error) {
	p := syscall.StringToUTF16(path)
	b := p // GetLongPathName says we can reuse buffer
	n, err := syscall.GetLongPathName(&p[0], &b[0], uint32(len(b)))
	if err != nil {
		return "", err
	}
	if n > uint32(len(b)) {
		b = make([]uint16, n)
		n, err = syscall.GetLongPathName(&p[0], &b[0], uint32(len(b)))
		if err != nil {
			return "", err
		}
	}
	b = b[:n]
	return Clean(syscall.UTF16ToString(b)), nil
}
