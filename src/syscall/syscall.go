// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

/*
 * Foundation of system call interface.
 */

func Syscall(trap int64, a1, a2, a3 int64) (r1, r2, err int64);
func	AddrToInt(b *byte) int64;

export Syscall
export AddrToInt


