// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains extra hooks for testing the go command.
// It is compiled into the Go binary only when building the
// test copy; it does not get compiled into the standard go
// command, so these testing hooks are not present in the
// go command that everyone uses.

// +build testgo

package load

import "os"

func init() {
	if v := os.Getenv("TESTGO_IS_GO_RELEASE"); v != "" {
		isGoRelease = v == "1"
	}
}
