// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

var (
	Stdin  = 0
	Stdout = 1
	Stderr = 2
)

func Syscall(trap, a1, a2, a3 uintptr) (r1, r2, err uintptr)
func Syscall6(trap, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, err uintptr)
func RawSyscall(trap, a1, a2, a3 uintptr) (r1, r2, err uintptr)
func RawSyscall6(trap, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, err uintptr)

func Errstr(errno int) string {
	if errno < 0 || errno >= int(len(errors)) {
		return "error " + itoa(errno)
	}
	return errors[errno]
}
