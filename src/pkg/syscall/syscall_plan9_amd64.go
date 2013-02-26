// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

func Getpagesize() int { return 0x200000 }

// Used by Gettimeofday, which expects
// an error return value.
func nanotime() (int64, error) {
	r1, _, _ := RawSyscall(SYS_NANOTIME, 0, 0, 0)
	return int64(r1), nil
}
