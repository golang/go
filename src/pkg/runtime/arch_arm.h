// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

enum {
	thechar = '5',
	BigEndian = 0,
	CacheLineSize = 32,
	RuntimeGogoBytes = 60,
#ifdef GOOS_nacl
	PhysPageSize = 65536,
#else
	PhysPageSize = 4096,
#endif
	PCQuantum = 4,
	Int64Align = 4
};
