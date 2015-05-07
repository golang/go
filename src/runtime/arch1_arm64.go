// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

const (
	thechar           = '7'
	_BigEndian        = 0
	_CacheLineSize    = 32
	_RuntimeGogoBytes = 64
	_PhysPageSize     = 4096*(1-goos_darwin) + 16384*goos_darwin
	_PCQuantum        = 4
	_Int64Align       = 8
	hugePageSize      = 0
)
