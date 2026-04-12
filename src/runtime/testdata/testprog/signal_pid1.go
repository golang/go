// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"
	"time"
)

func init() {
	register("SignalPid1", SignalPid1)
}

// SignalPid1 is a helper for TestSignalPid1.
func SignalPid1() {
	if os.Getpid() != 1 {
		fmt.Fprintln(os.Stderr, "I am not PID 1")
		return
	}
	fmt.Println("ready")

	time.Sleep(time.Hour)
}
