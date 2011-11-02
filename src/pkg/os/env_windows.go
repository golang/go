// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Windows environment variables.

package os

import (
	"errors"
	"syscall"
	"utf16"
	"unsafe"
)

// ENOENV is the error indicating that an environment variable does not exist.
var ENOENV = errors.New("no such environment variable")

// Getenverror retrieves the value of the environment variable named by the key.
// It returns the value and an error, if any.
func Getenverror(key string) (value string, err error) {
	b := make([]uint16, 100)
	n, e := syscall.GetEnvironmentVariable(syscall.StringToUTF16Ptr(key), &b[0], uint32(len(b)))
	if n == 0 && e == syscall.ERROR_ENVVAR_NOT_FOUND {
		return "", ENOENV
	}
	if n > uint32(len(b)) {
		b = make([]uint16, n)
		n, e = syscall.GetEnvironmentVariable(syscall.StringToUTF16Ptr(key), &b[0], uint32(len(b)))
		if n > uint32(len(b)) {
			n = 0
		}
	}
	if n == 0 {
		return "", NewSyscallError("GetEnvironmentVariable", e)
	}
	return string(utf16.Decode(b[0:n])), nil
}

// Getenv retrieves the value of the environment variable named by the key.
// It returns the value, which will be empty if the variable is not present.
func Getenv(key string) string {
	v, _ := Getenverror(key)
	return v
}

// Setenv sets the value of the environment variable named by the key.
// It returns an error, if any.
func Setenv(key, value string) error {
	var v *uint16
	if len(value) > 0 {
		v = syscall.StringToUTF16Ptr(value)
	}
	e := syscall.SetEnvironmentVariable(syscall.StringToUTF16Ptr(key), v)
	if e != 0 {
		return NewSyscallError("SetEnvironmentVariable", e)
	}
	return nil
}

// Clearenv deletes all environment variables.
func Clearenv() {
	for _, s := range Environ() {
		// Environment variables can begin with =
		// so start looking for the separator = at j=1.
		// http://blogs.msdn.com/b/oldnewthing/archive/2010/05/06/10008132.aspx
		for j := 1; j < len(s); j++ {
			if s[j] == '=' {
				Setenv(s[0:j], "")
				break
			}
		}
	}
}

// Environ returns an array of strings representing the environment,
// in the form "key=value".
func Environ() []string {
	s, e := syscall.GetEnvironmentStrings()
	if e != 0 {
		return nil
	}
	defer syscall.FreeEnvironmentStrings(s)
	r := make([]string, 0, 50) // Empty with room to grow.
	for from, i, p := 0, 0, (*[1 << 24]uint16)(unsafe.Pointer(s)); true; i++ {
		if p[i] == 0 {
			// empty string marks the end
			if i <= from {
				break
			}
			r = append(r, string(utf16.Decode(p[from:i])))
			from = i + 1
		}
	}
	return r
}

// TempDir returns the default directory to use for temporary files.
func TempDir() string {
	const pathSep = '\\'
	dirw := make([]uint16, syscall.MAX_PATH)
	n, _ := syscall.GetTempPath(uint32(len(dirw)), &dirw[0])
	if n > uint32(len(dirw)) {
		dirw = make([]uint16, n)
		n, _ = syscall.GetTempPath(uint32(len(dirw)), &dirw[0])
		if n > uint32(len(dirw)) {
			n = 0
		}
	}
	if n > 0 && dirw[n-1] == pathSep {
		n--
	}
	return string(utf16.Decode(dirw[0:n]))
}

func init() {
	var argc int32
	cmd := syscall.GetCommandLine()
	argv, e := syscall.CommandLineToArgv(cmd, &argc)
	if e != 0 {
		return
	}
	defer syscall.LocalFree(syscall.Handle(uintptr(unsafe.Pointer(argv))))
	Args = make([]string, argc)
	for i, v := range (*argv)[:argc] {
		Args[i] = string(syscall.UTF16ToString((*v)[:]))
	}
}
