// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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
	case *SyscallError:
		err = pe.Err
	}
	return err == syscall.ERROR_ALREADY_EXISTS ||
		err == syscall.ERROR_FILE_EXISTS || err == ErrExist
}

const _ERROR_BAD_NETPATH = syscall.Errno(53)

func isNotExist(err error) bool {
	switch pe := err.(type) {
	case nil:
		return false
	case *PathError:
		err = pe.Err
	case *LinkError:
		err = pe.Err
	case *SyscallError:
		err = pe.Err
	}
	return err == syscall.ERROR_FILE_NOT_FOUND ||
		err == _ERROR_BAD_NETPATH ||
		err == syscall.ERROR_PATH_NOT_FOUND || err == ErrNotExist
}

func isPermission(err error) bool {
	switch pe := err.(type) {
	case nil:
		return false
	case *PathError:
		err = pe.Err
	case *LinkError:
		err = pe.Err
	case *SyscallError:
		err = pe.Err
	}
	return err == syscall.ERROR_ACCESS_DENIED || err == ErrPermission
}
