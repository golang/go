// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

import "testing"

// The actual test functions are in non-_test.go files
// so that they can use cgo (import "C").
// These wrappers are here for gotest to find.

func TestAlign(t *testing.T)                 { testAlign(t) }
func TestConst(t *testing.T)                 { testConst(t) }
func TestEnum(t *testing.T)                  { testEnum(t) }
func TestAtol(t *testing.T)                  { testAtol(t) }
func TestErrno(t *testing.T)                 { testErrno(t) }
func TestMultipleAssign(t *testing.T)        { testMultipleAssign(t) }
func TestUnsignedInt(t *testing.T)           { testUnsignedInt(t) }
func TestCallback(t *testing.T)              { testCallback(t) }
func TestCallbackGC(t *testing.T)            { testCallbackGC(t) }
func TestCallbackPanic(t *testing.T)         { testCallbackPanic(t) }
func TestCallbackPanicLoop(t *testing.T)     { testCallbackPanicLoop(t) }
func TestCallbackPanicLocked(t *testing.T)   { testCallbackPanicLocked(t) }
func TestPanicFromC(t *testing.T)            { testPanicFromC(t) }
func TestZeroArgCallback(t *testing.T)       { testZeroArgCallback(t) }
func TestBlocking(t *testing.T)              { testBlocking(t) }
func Test1328(t *testing.T)                  { test1328(t) }
func TestParallelSleep(t *testing.T)         { testParallelSleep(t) }
func TestSetEnv(t *testing.T)                { testSetEnv(t) }
func TestHelpers(t *testing.T)               { testHelpers(t) }
func TestLibgcc(t *testing.T)                { testLibgcc(t) }
func Test1635(t *testing.T)                  { test1635(t) }
func TestPrintf(t *testing.T)                { testPrintf(t) }
func Test4029(t *testing.T)                  { test4029(t) }
func TestBoolAlign(t *testing.T)             { testBoolAlign(t) }
func Test3729(t *testing.T)                  { test3729(t) }
func Test3775(t *testing.T)                  { test3775(t) }
func TestCthread(t *testing.T)               { testCthread(t) }
func TestCallbackCallers(t *testing.T)       { testCallbackCallers(t) }
func Test5227(t *testing.T)                  { test5227(t) }
func TestCflags(t *testing.T)                { testCflags(t) }
func Test5337(t *testing.T)                  { test5337(t) }
func Test5548(t *testing.T)                  { test5548(t) }
func Test5603(t *testing.T)                  { test5603(t) }
func Test6833(t *testing.T)                  { test6833(t) }
func Test3250(t *testing.T)                  { test3250(t) }
func TestCallbackStack(t *testing.T)         { testCallbackStack(t) }
func TestFpVar(t *testing.T)                 { testFpVar(t) }
func Test4339(t *testing.T)                  { test4339(t) }
func Test6390(t *testing.T)                  { test6390(t) }
func Test5986(t *testing.T)                  { test5986(t) }
func Test7665(t *testing.T)                  { test7665(t) }
func TestNaming(t *testing.T)                { testNaming(t) }
func Test7560(t *testing.T)                  { test7560(t) }
func Test5242(t *testing.T)                  { test5242(t) }
func Test8092(t *testing.T)                  { test8092(t) }
func Test7978(t *testing.T)                  { test7978(t) }
func Test8694(t *testing.T)                  { test8694(t) }
func Test8517(t *testing.T)                  { test8517(t) }
func Test8811(t *testing.T)                  { test8811(t) }
func TestReturnAfterGrow(t *testing.T)       { testReturnAfterGrow(t) }
func TestReturnAfterGrowFromGo(t *testing.T) { testReturnAfterGrowFromGo(t) }
func Test9026(t *testing.T)                  { test9026(t) }
func Test9510(t *testing.T)                  { test9510(t) }
func Test9557(t *testing.T)                  { test9557(t) }
func Test10303(t *testing.T)                 { test10303(t, 10) }
func Test11925(t *testing.T)                 { test11925(t) }
func Test12030(t *testing.T)                 { test12030(t) }
func TestGCC68255(t *testing.T)              { testGCC68255(t) }
func TestCallGoWithString(t *testing.T)      { testCallGoWithString(t) }
func Test14838(t *testing.T)                 { test14838(t) }

func BenchmarkCgoCall(b *testing.B) { benchCgoCall(b) }
