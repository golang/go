// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

const (
	thechar        = '0'
	_BigEndian     = 1
	_CacheLineSize = 32
	_PhysPageSize  = 16384
	_PCQuantum     = 4
	_Int64Align    = 8
	hugePageSize   = 0
	minFrameSize   = 8
)

type uintreg uint64
type intptr int64 // TODO(rsc): remove
