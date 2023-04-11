// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || solaris
// +build aix solaris

package unix

import (
	"unsafe"
)

// ioctl itself should not be exposed directly, but additional get/set
// functions for specific types are permissible.

// IoctlSetInt performs an ioctl operation which sets an integer value
// on fd, using the specified request number.
func IoctlSetInt(fd int, req int, value int) error {
	return ioctl(fd, req, uintptr(value))
}

// IoctlSetPointerInt performs an ioctl operation which sets an
// integer value on fd, using the specified request number. The ioctl
// argument is called with a pointer to the integer value, rather than
// passing the integer value directly.
func IoctlSetPointerInt(fd int, req int, value int) error {
	v := int32(value)
	return ioctlPtr(fd, req, unsafe.Pointer(&v))
}

// IoctlSetWinsize performs an ioctl on fd with a *Winsize argument.
//
// To change fd's window size, the req argument should be TIOCSWINSZ.
func IoctlSetWinsize(fd int, req int, value *Winsize) error {
	// TODO: if we get the chance, remove the req parameter and
	// hardcode TIOCSWINSZ.
	return ioctlPtr(fd, req, unsafe.Pointer(value))
}

// IoctlSetTermios performs an ioctl on fd with a *Termios.
//
// The req value will usually be TCSETA or TIOCSETA.
func IoctlSetTermios(fd int, req int, value *Termios) error {
	// TODO: if we get the chance, remove the req parameter.
	return ioctlPtr(fd, req, unsafe.Pointer(value))
}

// IoctlGetInt performs an ioctl operation which gets an integer value
// from fd, using the specified request number.
//
// A few ioctl requests use the return value as an output parameter;
// for those, IoctlRetInt should be used instead of this function.
func IoctlGetInt(fd int, req int) (int, error) {
	var value int
	err := ioctlPtr(fd, req, unsafe.Pointer(&value))
	return value, err
}

func IoctlGetWinsize(fd int, req int) (*Winsize, error) {
	var value Winsize
	err := ioctlPtr(fd, req, unsafe.Pointer(&value))
	return &value, err
}

func IoctlGetTermios(fd int, req int) (*Termios, error) {
	var value Termios
	err := ioctlPtr(fd, req, unsafe.Pointer(&value))
	return &value, err
}
