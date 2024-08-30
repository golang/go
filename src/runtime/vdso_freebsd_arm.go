// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

const (
	_VDSO_TH_ALGO_ARM_GENTIM = 1
)

func getCntxct(physical bool) uint32

//go:nosplit
func (th *vdsoTimehands) getTimecounter() (uint32, bool) {
	switch th.algo {
	case _VDSO_TH_ALGO_ARM_GENTIM:
		return getCntxct(th.physical != 0), true
	default:
		return 0, false
	}
}
