// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.23
// +build go1.23

package crashmonitor

import "runtime/debug"

func init() {
	setCrashOutput = debug.SetCrashOutput
}
