// runindir

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 11656: runtime: jump to bad PC missing good traceback

// windows doesn't work, because Windows exception handling
// delivers signals based on the current PC, and that current PC
// doesn't go into the Go runtime.

// wasm does not work, because the linear memory is not executable.

// This test doesn't work on gccgo/GoLLVM, because they will not find
// any unwind information for the artificial function, and will not be
// able to unwind past that point.

//go:build !windows && !wasm && !gccgo

package ignored
