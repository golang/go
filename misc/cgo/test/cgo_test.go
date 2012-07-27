// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

import "testing"

// The actual test functions are in non-_test.go files 
// so that they can use cgo (import "C").
// These wrappers are here for gotest to find.

func TestAlign(t *testing.T)               { testAlign(t) }
func TestConst(t *testing.T)               { testConst(t) }
func TestEnum(t *testing.T)                { testEnum(t) }
func TestAtol(t *testing.T)                { testAtol(t) }
func TestErrno(t *testing.T)               { testErrno(t) }
func TestMultipleAssign(t *testing.T)      { testMultipleAssign(t) }
func TestCallback(t *testing.T)            { testCallback(t) }
func TestCallbackGC(t *testing.T)          { testCallbackGC(t) }
func TestCallbackPanic(t *testing.T)       { testCallbackPanic(t) }
func TestCallbackPanicLoop(t *testing.T)   { testCallbackPanicLoop(t) }
func TestCallbackPanicLocked(t *testing.T) { testCallbackPanicLocked(t) }
func TestZeroArgCallback(t *testing.T)     { testZeroArgCallback(t) }
func TestBlocking(t *testing.T)            { testBlocking(t) }
func Test1328(t *testing.T)                { test1328(t) }
func TestParallelSleep(t *testing.T)       { testParallelSleep(t) }
func TestSetEnv(t *testing.T)              { testSetEnv(t) }
func TestHelpers(t *testing.T)             { testHelpers(t) }
func TestSetgid(t *testing.T)              { testSetgid(t) }

func BenchmarkCgoCall(b *testing.B) { benchCgoCall(b) }
