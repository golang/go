// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd solaris

package os

import "syscall"

func isExist(err error) bool {
	switch pe := err.(type) {
	case nil:
		return false
	case *PathError:
		err = pe.Err
	case *LinkError:
		err = pe.Err
	}
	return err == syscall.EEXIST || err == ErrExist
}

func isNotExist(err error) bool {
	switch pe := err.(type) {
	case nil:
		return false
	case *PathError:
		err = pe.Err
	case *LinkError:
		err = pe.Err
	}
	return err == syscall.ENOENT || err == ErrNotExist
}

func isPermission(err error) bool {
	switch pe := err.(type) {
	case nil:
		return false
	case *PathError:
		err = pe.Err
	case *LinkError:
		err = pe.Err
	}
	return err == syscall.EACCES || err == syscall.EPERM || err == ErrPermission
}
