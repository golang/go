// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Expose some runtime functions for testing.
// This is the gccgo version of runtime.c.

// +build gccgo

_Bool runtime_lockedOSThread(void);

_Bool LockedOSThread(void) asm(GOPKGPATH ".LockedOSThread");

_Bool
LockedOSThread(void)
{
	return runtime_lockedOSThread();
}
