// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unix

import "unsafe"

// FcntlInt performs a fcntl syscall on fd with the provided command and argument.
func FcntlInt(fd uintptr, cmd, arg int) (int, error) {
	return fcntl(int(fd), cmd, arg)
}

// FcntlFlock performs a fcntl syscall for the F_GETLK, F_SETLK or F_SETLKW command.
func FcntlFlock(fd uintptr, cmd int, lk *Flock_t) error {
	_, err := fcntl(int(fd), cmd, int(uintptr(unsafe.Pointer(lk))))
	return err
}

// FcntlFstore performs a fcntl syscall for the F_PREALLOCATE command.
func FcntlFstore(fd uintptr, cmd int, fstore *Fstore_t) error {
	_, err := fcntl(int(fd), cmd, int(uintptr(unsafe.Pointer(fstore))))
	return err
}
