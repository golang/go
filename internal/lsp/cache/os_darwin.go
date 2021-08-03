// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"unsafe"
)

func init() {
	checkPathCase = darwinCheckPathCase
}

func darwinCheckPathCase(path string) error {
	// Darwin provides fcntl(F_GETPATH) to get a path for an arbitrary FD.
	// Conveniently for our purposes, it gives the canonical case back. But
	// there's no guarantee that it will follow the same route through the
	// filesystem that the original path did.

	path, err := filepath.Abs(path)
	if err != nil {
		return err
	}
	fd, err := syscall.Open(path, os.O_RDONLY, 0)
	if err != nil {
		return err
	}
	defer syscall.Close(fd)
	buf := make([]byte, 4096) // No MAXPATHLEN in syscall, I think it's 1024, this is bigger.

	// Wheeee! syscall doesn't expose a way to call Fcntl except FcntlFlock.
	// As of writing, it just passes the pointer through, so we can just lie.
	if err := syscall.FcntlFlock(uintptr(fd), syscall.F_GETPATH, (*syscall.Flock_t)(unsafe.Pointer(&buf[0]))); err != nil {
		return err
	}
	buf = buf[:bytes.IndexByte(buf, 0)]

	isRoot := func(p string) bool {
		return p[len(p)-1] == filepath.Separator
	}
	// Darwin seems to like having multiple names for the same folder. Match as much of the suffix as we can.
	for got, want := path, string(buf); !isRoot(got) && !isRoot(want); got, want = filepath.Dir(got), filepath.Dir(want) {
		g, w := filepath.Base(got), filepath.Base(want)
		if !strings.EqualFold(g, w) {
			break
		}
		if g != w {
			return fmt.Errorf("case mismatch in path %q: component %q is listed by macOS as %q", path, g, w)
		}
	}
	return nil
}
