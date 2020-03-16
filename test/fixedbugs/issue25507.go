// errorcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// We have a limit of 1GB for stack frames.
// Test that we extend that limit to include large argument/return areas.
// Argument/return areas are part of the parent frame, not the frame itself,
// so they need to be handled separately.

package main

// >1GB to trigger failure, <2GB to work on 32-bit platforms.
type large struct {
	b [1500000000]byte
}

func (x large) f1() int { // ERROR "stack frame too large"
	return 5
}

func f2(x large) int { // ERROR "stack frame too large"
	return 5
}

func f3() (x large, i int) { // ERROR "stack frame too large"
	return
}
