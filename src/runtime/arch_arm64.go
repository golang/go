// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

const (
	thechar        = '7'
	_BigEndian     = 0
	_CacheLineSize = 32
	_PhysPageSize  = 65536
	_PCQuantum     = 4
	_Int64Align    = 8
	hugePageSize   = 0
)

type uintreg uint64
type intptr int64 // TODO(rsc): remove
