// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

const (
	thechar        = '6'
	_BigEndian     = 0
	_CacheLineSize = 64
	_PhysPageSize  = 65536*goos_nacl + 4096*(1-goos_nacl)
	_PCQuantum     = 1
	_Int64Align    = 8
	hugePageSize   = 1 << 21
)
