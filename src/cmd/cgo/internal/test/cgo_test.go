// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build cgo

package cgotest

import "testing"

// The actual test functions are in non-_test.go files
// so that they can use cgo (import "C").
// These wrappers are here for gotest to find.

func Test1328(t *testing.T)                  { test1328(t) }
func Test1560(t *testing.T)                  { test1560(t) }
func Test1635(t *testing.T)                  { test1635(t) }
func Test3250(t *testing.T)                  { test3250(t) }
func Test3729(t *testing.T)                  { test3729(t) }
func Test3775(t *testing.T)                  { test3775(t) }
func Test4029(t *testing.T)                  { test4029(t) }
func Test4339(t *testing.T)                  { test4339(t) }
func Test5227(t *testing.T)                  { test5227(t) }
func Test5242(t *testing.T)                  { test5242(t) }
func Test5337(t *testing.T)                  { test5337(t) }
func Test5548(t *testing.T)                  { test5548(t) }
func Test5603(t *testing.T)                  { test5603(t) }
func Test5986(t *testing.T)                  { test5986(t) }
func Test6390(t *testing.T)                  { test6390(t) }
func Test6833(t *testing.T)                  { test6833(t) }
func Test6907(t *testing.T)                  { test6907(t) }
func Test6907Go(t *testing.T)                { test6907Go(t) }
func Test7560(t *testing.T)                  { test7560(t) }
func Test7665(t *testing.T)                  { test7665(t) }
func Test7978(t *testing.T)                  { test7978(t) }
func Test8092(t *testing.T)                  { test8092(t) }
func Test8517(t *testing.T)                  { test8517(t) }
func Test8694(t *testing.T)                  { test8694(t) }
func Test8756(t *testing.T)                  { test8756(t) }
func Test8811(t *testing.T)                  { test8811(t) }
func Test9026(t *testing.T)                  { test9026(t) }
func Test9510(t *testing.T)                  { test9510(t) }
func Test9557(t *testing.T)                  { test9557(t) }
func Test10303(t *testing.T)                 { test10303(t, 10) }
func Test11925(t *testing.T)                 { test11925(t) }
func Test12030(t *testing.T)                 { test12030(t) }
func Test14838(t *testing.T)                 { test14838(t) }
func Test17065(t *testing.T)                 { test17065(t) }
func Test17537(t *testing.T)                 { test17537(t) }
func Test18126(t *testing.T)                 { test18126(t) }
func Test18720(t *testing.T)                 { test18720(t) }
func Test20129(t *testing.T)                 { test20129(t) }
func Test20266(t *testing.T)                 { test20266(t) }
func Test20369(t *testing.T)                 { test20369(t) }
func Test20910(t *testing.T)                 { test20910(t) }
func Test21708(t *testing.T)                 { test21708(t) }
func Test21809(t *testing.T)                 { test21809(t) }
func Test21897(t *testing.T)                 { test21897(t) }
func Test22906(t *testing.T)                 { test22906(t) }
func Test23356(t *testing.T)                 { test23356(t) }
func Test24206(t *testing.T)                 { test24206(t) }
func Test25143(t *testing.T)                 { test25143(t) }
func Test26066(t *testing.T)                 { test26066(t) }
func Test26213(t *testing.T)                 { test26213(t) }
func Test27660(t *testing.T)                 { test27660(t) }
func Test28896(t *testing.T)                 { test28896(t) }
func Test30065(t *testing.T)                 { test30065(t) }
func Test32579(t *testing.T)                 { test32579(t) }
func Test31891(t *testing.T)                 { test31891(t) }
func Test42018(t *testing.T)                 { test42018(t) }
func Test45451(t *testing.T)                 { test45451(t) }
func Test49633(t *testing.T)                 { test49633(t) }
func Test69086(t *testing.T)                 { test69086(t) }
func TestAlign(t *testing.T)                 { testAlign(t) }
func TestAtol(t *testing.T)                  { testAtol(t) }
func TestBlocking(t *testing.T)              { testBlocking(t) }
func TestBoolAlign(t *testing.T)             { testBoolAlign(t) }
func TestCallGoWithString(t *testing.T)      { testCallGoWithString(t) }
func TestCallback(t *testing.T)              { testCallback(t) }
func TestCallbackCallers(t *testing.T)       { testCallbackCallers(t) }
func TestCallbackGC(t *testing.T)            { testCallbackGC(t) }
func TestCallbackPanic(t *testing.T)         { testCallbackPanic(t) }
func TestCallbackPanicLocked(t *testing.T)   { testCallbackPanicLocked(t) }
func TestCallbackPanicLoop(t *testing.T)     { testCallbackPanicLoop(t) }
func TestCallbackStack(t *testing.T)         { testCallbackStack(t) }
func TestCflags(t *testing.T)                { testCflags(t) }
func TestCheckConst(t *testing.T)            { testCheckConst(t) }
func TestConst(t *testing.T)                 { testConst(t) }
func TestCthread(t *testing.T)               { testCthread(t) }
func TestEnum(t *testing.T)                  { testEnum(t) }
func TestNamedEnum(t *testing.T)             { testNamedEnum(t) }
func TestCastToEnum(t *testing.T)            { testCastToEnum(t) }
func TestErrno(t *testing.T)                 { testErrno(t) }
func TestFpVar(t *testing.T)                 { testFpVar(t) }
func TestGCC68255(t *testing.T)              { testGCC68255(t) }
func TestHandle(t *testing.T)                { testHandle(t) }
func TestHelpers(t *testing.T)               { testHelpers(t) }
func TestLibgcc(t *testing.T)                { testLibgcc(t) }
func TestMultipleAssign(t *testing.T)        { testMultipleAssign(t) }
func TestNaming(t *testing.T)                { testNaming(t) }
func TestPanicFromC(t *testing.T)            { testPanicFromC(t) }
func TestPrintf(t *testing.T)                { testPrintf(t) }
func TestReturnAfterGrow(t *testing.T)       { testReturnAfterGrow(t) }
func TestReturnAfterGrowFromGo(t *testing.T) { testReturnAfterGrowFromGo(t) }
func TestSetEnv(t *testing.T)                { testSetEnv(t) }
func TestThreadLock(t *testing.T)            { testThreadLockFunc(t) }
func TestUnsignedInt(t *testing.T)           { testUnsignedInt(t) }
func TestZeroArgCallback(t *testing.T)       { testZeroArgCallback(t) }

func BenchmarkCgoCall(b *testing.B)      { benchCgoCall(b) }
func BenchmarkGoString(b *testing.B)     { benchGoString(b) }
func BenchmarkCGoCallback(b *testing.B)  { benchCallback(b) }
func BenchmarkCGoInCThread(b *testing.B) { benchCGoInCthread(b) }
