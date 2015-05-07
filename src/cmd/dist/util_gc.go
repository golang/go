// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !gccgo

package main

func cpuid(info *[4]uint32, ax uint32)

func cansse2() bool {
	if gohostarch != "386" && gohostarch != "amd64" {
		return false
	}

	var info [4]uint32
	cpuid(&info, 1)
	return info[3]&(1<<26) != 0 // SSE2
}
