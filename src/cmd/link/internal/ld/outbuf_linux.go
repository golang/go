// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import "syscall"

func (out *OutBuf) fallocate(size uint64) error {
	return syscall.Fallocate(int(out.f.Fd()), 0, 0, int64(size))
}
