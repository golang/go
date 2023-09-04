// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix && !android && !openbsd

package main

/*
void callStackSwitchCallbackFromThread(void);
*/
import "C"

import (
	"fmt"
	"runtime/debug"
)

func init() {
	register("StackSwitchCallback", StackSwitchCallback)
}

//export stackSwitchCallback
func stackSwitchCallback() {
	// We want to trigger a bounds check on the g0 stack. To do this, we
	// need to call a splittable function through systemstack().
	// SetGCPercent contains such a systemstack call.
	gogc := debug.SetGCPercent(100)
	debug.SetGCPercent(gogc)
}


// Regression test for https://go.dev/issue/62440. It should be possible for C
// threads to call into Go from different stacks without crashing due to g0
// stack bounds checks.
//
// N.B. This is only OK for threads created in C. Threads with Go frames up the
// stack must not change the stack out from under us.
func StackSwitchCallback() {
	C.callStackSwitchCallbackFromThread();

	fmt.Printf("OK\n")
}
