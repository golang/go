// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore
// This test depends on running C code on Go stacks. Not allowed anymore.

// Demo of deferred C function with untrue prototype
// breaking stack copying. See golang.org/issue/7695.

package cgotest

import (
	"testing"

	"./backdoor"
)

func TestIssue7695(t *testing.T) {
	defer backdoor.Issue7695(1, 0, 2, 0, 0, 3, 0, 4)
	recurse(100)
}

func recurse(n int) {
	var x [128]int
	n += x[0]
	if n > 0 {
		recurse(n - 1)
	}
}
