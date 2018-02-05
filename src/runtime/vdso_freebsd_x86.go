// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build freebsd
// +build 386 amd64

package runtime

const (
	_VDSO_TH_ALGO_X86_TSC = 1
)

//go:nosplit
func (th *vdsoTimehands) getTSCTimecounter() uint32 {
	tsc := cputicks()
	if th.x86_shift > 0 {
		tsc >>= th.x86_shift
	}
	return uint32(tsc)
}

//go:nosplit
func (th *vdsoTimehands) getTimecounter() (uint32, bool) {
	switch th.algo {
	case _VDSO_TH_ALGO_X86_TSC:
		return th.getTSCTimecounter(), true
	default:
		return 0, false
	}
}
