// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !darwin && !(freebsd && go1.21) && !linux

package ld

func (out *OutBuf) fallocate(size uint64) error {
	return errNoFallocate
}
