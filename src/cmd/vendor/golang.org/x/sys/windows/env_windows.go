// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Windows environment variables.

package windows

import (
	"syscall"
	"unicode/utf16"
	"unsafe"
)

func Getenv(key string) (value string, found bool) {
	return syscall.Getenv(key)
}

func Setenv(key, value string) error {
	return syscall.Setenv(key, value)
}

func Clearenv() {
	syscall.Clearenv()
}

func Environ() []string {
	return syscall.Environ()
}

// Returns a default environment associated with the token, rather than the current
// process. If inheritExisting is true, then this environment also inherits the
// environment of the current process.
func (token Token) Environ(inheritExisting bool) (env []string, err error) {
	var block *uint16
	err = CreateEnvironmentBlock(&block, token, inheritExisting)
	if err != nil {
		return nil, err
	}
	defer DestroyEnvironmentBlock(block)
	blockp := uintptr(unsafe.Pointer(block))
	for {
		entry := (*[(1 << 30) - 1]uint16)(unsafe.Pointer(blockp))[:]
		for i, v := range entry {
			if v == 0 {
				entry = entry[:i]
				break
			}
		}
		if len(entry) == 0 {
			break
		}
		env = append(env, string(utf16.Decode(entry)))
		blockp += 2 * (uintptr(len(entry)) + 1)
	}
	return env, nil
}

func Unsetenv(key string) error {
	return syscall.Unsetenv(key)
}
