// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

const (
	_VDSO_TH_ALGO_RISCV_RDTIME = 1
)

func getCntxct() uint32

//go:nosplit
func (th *vdsoTimehands) getTimecounter() (uint32, bool) {
	switch th.algo {
	case _VDSO_TH_ALGO_RISCV_RDTIME:
		return getCntxct(), true
	default:
		return 0, false
	}
}
