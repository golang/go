// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package debug

import (
	"time"
)

// Uses assembly to call corresponding runtime-internal functions.
func setMaxStack(int) int
func setGCPercent(int32) int32
func setPanicOnFault(bool) bool
func setMaxThreads(int) int

// Implemented in package runtime.
func readGCStats(*[]time.Duration)
func enableGC(bool) bool
func freeOSMemory()
