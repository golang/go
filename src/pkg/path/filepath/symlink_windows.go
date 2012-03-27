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
	s := syscall.UTF16ToString(b)
	// syscall.GetLongPathName does not change the case of the drive letter,
	// but the result of EvalSymlinks must be unique, so we have
	// EvalSymlinks(`c:\a`) == EvalSymlinks(`C:\a`).
	// Make drive letter upper case. This matches what os.Getwd returns.
	if len(s) >= 2 && s[1] == ':' && 'a' <= s[0] && s[0] <= 'z' {
		s = string(s[0]+'A'-'a') + s[1:]
	}
	return Clean(s), nil
}
