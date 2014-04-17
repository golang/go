// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package backdoor

func LockedOSThread() bool // in runtime.c
func Issue7695(x1, x2, x3, x4, x5, x6, x7, x8 uintptr)
