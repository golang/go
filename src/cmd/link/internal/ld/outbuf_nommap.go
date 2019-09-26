// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !darwin,!dragonfly,!freebsd,!linux,!openbsd,!windows

package ld

import "errors"

var errNotSupported = errors.New("mmap not supported")

func (out *OutBuf) Mmap(filesize uint64) error { return errNotSupported }
func (out *OutBuf) Munmap()                    { panic("unreachable") }
func (out *OutBuf) Msync() error               { panic("unreachable") }
