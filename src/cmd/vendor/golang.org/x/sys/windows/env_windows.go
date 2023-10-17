// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Windows environment variables.

package windows

import (
	"syscall"
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
	blockp := unsafe.Pointer(block)
	for {
		entry := UTF16PtrToString((*uint16)(blockp))
		if len(entry) == 0 {
			break
		}
		env = append(env, entry)
		blockp = unsafe.Add(blockp, 2*(len(entry)+1))
	}
	return env, nil
}

func Unsetenv(key string) error {
	return syscall.Unsetenv(key)
}
