// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

enum {
	thechar = '6',
	BigEndian = 0,
	CacheLineSize = 64,
#ifdef GOOS_solaris
	RuntimeGogoBytes = 80,
#else
#ifdef GOOS_windows
	RuntimeGogoBytes = 80,
#else
#ifdef GOOS_plan9
	RuntimeGogoBytes = 80,
#else
	RuntimeGogoBytes = 64,
#endif	// Plan 9
#endif	// Windows
#endif	// Solaris
	PhysPageSize = 4096,
	PCQuantum = 1,
	Int64Align = 8
};
