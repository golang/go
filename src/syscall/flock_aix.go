// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

import "unsafe"

// On AIX, there is no flock() system call, we emulate it.
// Moreover, we can't call the default fcntl syscall because the arguments
// must be integer and it's not possible to transform a pointer (lk)
// to a int value.
// It's easier to call syscall6 than to transform fcntl for every GOOS.
func fcntlFlock(fd, cmd int, lk *Flock_t) (err error) {
	_, _, e1 := syscall6(uintptr(unsafe.Pointer(&libc_fcntl)), 3, uintptr(fd), uintptr(cmd), uintptr(unsafe.Pointer(lk)), 0, 0, 0)
	if e1 != 0 {
		err = errnoErr(e1)
	}
	return
}

func Flock(fd int, op int) (err error) {
	lk := &Flock_t{}
	if (op & LOCK_UN) != 0 {
		lk.Type = F_UNLCK
	} else if (op & LOCK_EX) != 0 {
		lk.Type = F_WRLCK
	} else if (op & LOCK_SH) != 0 {
		lk.Type = F_RDLCK
	} else {
		return nil
	}
	if (op & LOCK_NB) != 0 {
		err = fcntlFlock(fd, F_SETLK, lk)
		if err != nil && (err == EAGAIN || err == EACCES) {
			return EWOULDBLOCK
		}
		return err
	}
	return fcntlFlock(fd, F_SETLKW, lk)
}
