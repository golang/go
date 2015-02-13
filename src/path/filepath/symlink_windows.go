// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package filepath

import (
	"syscall"
)

func toShort(path string) (string, error) {
	p, err := syscall.UTF16FromString(path)
	if err != nil {
		return "", err
	}
	b := p // GetShortPathName says we can reuse buffer
	n := uint32(len(b))
	for {
		n, err = syscall.GetShortPathName(&p[0], &b[0], uint32(len(b)))
		if err != nil {
			return "", err
		}
		if n <= uint32(len(b)) {
			return syscall.UTF16ToString(b[:n]), nil
		}
		b = make([]uint16, n)
	}
}

func toLong(path string) (string, error) {
	p, err := syscall.UTF16FromString(path)
	if err != nil {
		return "", err
	}
	b := p // GetLongPathName says we can reuse buffer
	n := uint32(len(b))
	for {
		n, err = syscall.GetLongPathName(&p[0], &b[0], uint32(len(b)))
		if err != nil {
			return "", err
		}
		if n <= uint32(len(b)) {
			return syscall.UTF16ToString(b[:n]), nil
		}
		b = make([]uint16, n)
	}
}

func evalSymlinks(path string) (string, error) {
	path, err := walkSymlinks(path)
	if err != nil {
		return "", err
	}

	p, err := toShort(path)
	if err != nil {
		return "", err
	}
	p, err = toLong(p)
	if err != nil {
		return "", err
	}
	// syscall.GetLongPathName does not change the case of the drive letter,
	// but the result of EvalSymlinks must be unique, so we have
	// EvalSymlinks(`c:\a`) == EvalSymlinks(`C:\a`).
	// Make drive letter upper case.
	if len(p) >= 2 && p[1] == ':' && 'a' <= p[0] && p[0] <= 'z' {
		p = string(p[0]+'A'-'a') + p[1:]
	}
	return Clean(p), nil
}
