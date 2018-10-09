// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the copylock checker's
// function declaration analysis.

package a

import "sync"

func OkFunc(*sync.Mutex) {}
func BadFunc(sync.Mutex) {} // want "BadFunc passes lock by value: sync.Mutex"
func BadFunc2(sync.Map)  {} // want "BadFunc2 passes lock by value: sync.Map contains sync.Mutex"
func OkRet() *sync.Mutex {}
func BadRet() sync.Mutex {} // Don't warn about results

var (
	OkClosure   = func(*sync.Mutex) {}
	BadClosure  = func(sync.Mutex) {} // want "func passes lock by value: sync.Mutex"
	BadClosure2 = func(sync.Map) {}   // want "func passes lock by value: sync.Map contains sync.Mutex"
)

type EmbeddedRWMutex struct {
	sync.RWMutex
}

func (*EmbeddedRWMutex) OkMeth() {}
func (EmbeddedRWMutex) BadMeth() {} // want "BadMeth passes lock by value: a.EmbeddedRWMutex"
func OkFunc(e *EmbeddedRWMutex)  {}
func BadFunc(EmbeddedRWMutex)    {} // want "BadFunc passes lock by value: a.EmbeddedRWMutex"
func OkRet() *EmbeddedRWMutex    {}
func BadRet() EmbeddedRWMutex    {} // Don't warn about results

type FieldMutex struct {
	s sync.Mutex
}

func (*FieldMutex) OkMeth()   {}
func (FieldMutex) BadMeth()   {} // want "BadMeth passes lock by value: a.FieldMutex contains sync.Mutex"
func OkFunc(*FieldMutex)      {}
func BadFunc(FieldMutex, int) {} // want "BadFunc passes lock by value: a.FieldMutex contains sync.Mutex"

type L0 struct {
	L1
}

type L1 struct {
	l L2
}

type L2 struct {
	sync.Mutex
}

func (*L0) Ok() {}
func (L0) Bad() {} // want "Bad passes lock by value: a.L0 contains a.L1 contains a.L2"

type EmbeddedMutexPointer struct {
	s *sync.Mutex // safe to copy this pointer
}

func (*EmbeddedMutexPointer) Ok()      {}
func (EmbeddedMutexPointer) AlsoOk()   {}
func StillOk(EmbeddedMutexPointer)     {}
func LookinGood() EmbeddedMutexPointer {}

type EmbeddedLocker struct {
	sync.Locker // safe to copy interface values
}

func (*EmbeddedLocker) Ok()    {}
func (EmbeddedLocker) AlsoOk() {}

type CustomLock struct{}

func (*CustomLock) Lock()   {}
func (*CustomLock) Unlock() {}

func Ok(*CustomLock) {}
func Bad(CustomLock) {} // want "Bad passes lock by value: a.CustomLock"

// Passing lock values into interface function arguments
func FuncCallInterfaceArg(f func(a int, b interface{})) {
	var m sync.Mutex
	var t struct{ lock sync.Mutex }

	f(1, "foo")
	f(2, &t)
	f(3, &sync.Mutex{})
	f(4, m) // want "call of f copies lock value: sync.Mutex"
	f(5, t) // want "call of f copies lock value: struct.lock sync.Mutex. contains sync.Mutex"
	var fntab []func(t)
	fntab[0](t) // want "call of fntab.0. copies lock value: struct.lock sync.Mutex. contains sync.Mutex"
}

// Returning lock via interface value
func ReturnViaInterface(x int) (int, interface{}) {
	var m sync.Mutex
	var t struct{ lock sync.Mutex }

	switch x % 4 {
	case 0:
		return 0, "qwe"
	case 1:
		return 1, &sync.Mutex{}
	case 2:
		return 2, m // want "return copies lock value: sync.Mutex"
	default:
		return 3, t // want "return copies lock value: struct.lock sync.Mutex. contains sync.Mutex"
	}
}

// Some cases that we don't warn about.

func AcceptedCases() {
	x := EmbeddedRwMutex{} // composite literal on RHS is OK (#16227)
	x = BadRet()           // function call on RHS is OK (#16227)
	x = *OKRet()           // indirection of function call on RHS is OK (#16227)
}

// TODO: Unfortunate cases

// Non-ideal error message:
// Since we're looking for Lock methods, sync.Once's underlying
// sync.Mutex gets called out, but without any reference to the sync.Once.
type LocalOnce sync.Once

func (LocalOnce) Bad() {} // want "Bad passes lock by value: a.LocalOnce contains sync.Mutex"

// False negative:
// LocalMutex doesn't have a Lock method.
// Nevertheless, it is probably a bad idea to pass it by value.
type LocalMutex sync.Mutex

func (LocalMutex) Bad() {} // WANTED: An error here :(
