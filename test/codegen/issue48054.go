// asmcheck

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

func a(n string) bool {
	// arm64:"CBZ"
	if len(n) > 0 {
		return true
	}
	return false
}

func a2(n []int) bool {
	// arm64:"CBZ"
	if len(n) > 0 {
		return true
	}
	return false
}

func a3(n []int) bool {
	// amd64:"TESTQ"
	if len(n) < 1 {
		return true
	}
	return false
}
