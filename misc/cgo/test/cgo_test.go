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
func TestUnsignedInt(t *testing.T)         { testUnsignedInt(t) }
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
func TestLibgcc(t *testing.T)              { testLibgcc(t) }
func Test1635(t *testing.T)                { test1635(t) }
func TestPrintf(t *testing.T)              { testPrintf(t) }
func Test4029(t *testing.T)                { test4029(t) }
func TestBoolAlign(t *testing.T)           { testBoolAlign(t) }
func Test3729(t *testing.T)                { test3729(t) }
func Test3775(t *testing.T)                { test3775(t) }
func TestCthread(t *testing.T)             { testCthread(t) }
func TestCallbackCallers(t *testing.T)     { testCallbackCallers(t) }
func Test5227(t *testing.T)                { test5227(t) }
func TestCflags(t *testing.T)              { testCflags(t) }
func Test5337(t *testing.T)                { test5337(t) }
func Test5548(t *testing.T)                { test5548(t) }
func Test5603(t *testing.T)                { test5603(t) }
func Test3250(t *testing.T)                { test3250(t) }
func TestCallbackStack(t *testing.T)       { testCallbackStack(t) }
func TestFpVar(t *testing.T)               { testFpVar(t) }
func Test4339(t *testing.T)                { test4339(t) }
func Test6390(t *testing.T)                { test6390(t) }
func Test5986(t *testing.T)                { test5986(t) }

func BenchmarkCgoCall(b *testing.B) { benchCgoCall(b) }
