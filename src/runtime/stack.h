// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// For the linkers. Must match Go definitions.
// TODO(rsc): Share Go definitions with linkers directly.

enum {
#ifdef GOOS_windows
	StackSystem = 512 * sizeof(uintptr),
#else
#ifdef GOOS_plan9
	StackSystem = 512,
#else
	StackSystem = 0,
#endif	// Plan 9
#endif	// Windows

	StackBig = 4096,
	StackGuard = 640 + StackSystem,
	StackSmall = 128,
	StackLimit = StackGuard - StackSystem - StackSmall,
};

#define StackPreempt ((uint64)-1314)
