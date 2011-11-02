// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Plan 9 environment variables.

package os

import (
	"error"
	"syscall"
)

// ENOENV is the error indicating that an environment variable does not exist.
var ENOENV = errors.New("no such environment variable")

// Getenverror retrieves the value of the environment variable named by the key.
// It returns the value and an error, if any.
func Getenverror(key string) (value string, err error) {
	if len(key) == 0 {
		return "", EINVAL
	}
	f, e := Open("/env/" + key)
	if iserror(e) {
		return "", ENOENV
	}
	defer f.Close()

	l, _ := f.Seek(0, 2)
	f.Seek(0, 0)
	buf := make([]byte, l)
	n, e := f.Read(buf)
	if iserror(e) {
		return "", ENOENV
	}

	if n > 0 && buf[n-1] == 0 {
		buf = buf[:n-1]
	}
	return string(buf), nil
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
	if len(key) == 0 {
		return EINVAL
	}

	f, e := Create("/env/" + key)
	if iserror(e) {
		return e
	}
	defer f.Close()

	_, e = f.Write([]byte(value))
	return nil
}

// Clearenv deletes all environment variables.
func Clearenv() {
	syscall.RawSyscall(syscall.SYS_RFORK, syscall.RFCENVG, 0, 0)
}

// Environ returns an array of strings representing the environment,
// in the form "key=value".
func Environ() []string {
	env := make([]string, 0, 100)

	f, e := Open("/env")
	if iserror(e) {
		panic(e)
	}
	defer f.Close()

	names, e := f.Readdirnames(-1)
	if iserror(e) {
		panic(e)
	}

	for _, k := range names {
		if v, e := Getenverror(k); !iserror(e) {
			env = append(env, k+"="+v)
		}
	}
	return env[0:len(env)]
}

// TempDir returns the default directory to use for temporary files.
func TempDir() string {
	return "/tmp"
}
