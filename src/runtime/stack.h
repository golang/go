// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// For the linkers. Must match Go definitions.
// TODO(rsc): Share Go definitions with linkers directly.

enum {
#ifdef GOOS_windows
#define STACKSYSTEM (512 * sizeof(uintptr))
#endif // GOOS_windows
#ifdef GOOS_plan9
#define STACKSYSTEM	512
#endif // GOOS_plan9
#ifdef GOOS_darwin
#ifdef GOARCH_arm
#define STACKSYSTEM 1024
#endif // GOARCH_arm
#endif // GOOS_darwin

#ifndef STACKSYSTEM
#define STACKSYSTEM 0
#endif

	StackSystem = STACKSYSTEM,

	StackBig = 4096,
	StackGuard = 640 + StackSystem,
	StackSmall = 128,
	StackLimit = StackGuard - StackSystem - StackSmall,
};

#define StackPreempt ((uint64)-1314)
/*c2go
enum
{
	StackPreempt = 1, // TODO: Change to (uint64)-1314 in Go translation
};
*/
