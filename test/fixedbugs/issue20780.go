// errorcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// We have a limit of 1GB for stack frames.
// Make sure we include the callee args section.
// (The dispatch wrapper which implements (*S).f
// copies the return value from f to a stack temp, then
// from that stack temp to the return value of (*S).f.
// It uses ~800MB for each section.)

package main

type S struct {
	i interface {
		f() [800e6]byte
	}
}
