// compile

// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that code compiles without
// "internal error: ... recorded as live on entry" errors
// from the liveness code.

package main

// The liveness analysis used to get confused by the tail return
// instruction in the wrapper methods generated for T1.M and (*T1).M,
// causing a spurious "live at entry: ~r1" for the return result.
// This test is checking that there is no such message.
// We cannot use live.go because it runs with -live on, which will
// generate (correct) messages about the wrapper's receivers
// being live on entry, but those messages correspond to no
// source line in the file, so they are given at line 1, which we
// cannot annotate. Not using -live here avoids that problem.

type T struct {
}

func (t *T) M() *int

type T1 struct {
	*T
}
