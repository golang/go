// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the copylock checker.

package testdata

import "sync"

func OkFunc(*sync.Mutex) {}
func BadFunc(sync.Mutex) {} // ERROR "BadFunc passes Lock by value: sync.Mutex"
func OkRet() *sync.Mutex {}
func BadRet() sync.Mutex {} // ERROR "BadRet returns Lock by value: sync.Mutex"

type EmbeddedRWMutex struct {
	sync.RWMutex
}

func (*EmbeddedRWMutex) OkMeth() {}
func (EmbeddedRWMutex) BadMeth() {} // ERROR "BadMeth passes Lock by value: testdata.EmbeddedRWMutex"
func OkFunc(e *EmbeddedRWMutex)  {}
func BadFunc(EmbeddedRWMutex)    {} // ERROR "BadFunc passes Lock by value: testdata.EmbeddedRWMutex"
func OkRet() *EmbeddedRWMutex    {}
func BadRet() EmbeddedRWMutex    {} // ERROR "BadRet returns Lock by value: testdata.EmbeddedRWMutex"

type FieldMutex struct {
	s sync.Mutex
}

func (*FieldMutex) OkMeth()   {}
func (FieldMutex) BadMeth()   {} // ERROR "BadMeth passes Lock by value: testdata.FieldMutex contains sync.Mutex"
func OkFunc(*FieldMutex)      {}
func BadFunc(FieldMutex, int) {} // ERROR "BadFunc passes Lock by value: testdata.FieldMutex contains sync.Mutex"

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
func (L0) Bad() {} // ERROR "Bad passes Lock by value: testdata.L0 contains testdata.L1 contains testdata.L2"

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
func Bad(CustomLock) {} // ERROR "Bad passes Lock by value: testdata.CustomLock"

// TODO: Unfortunate cases

// Non-ideal error message:
// Since we're looking for Lock methods, sync.Once's underlying
// sync.Mutex gets called out, but without any reference to the sync.Once.
type LocalOnce sync.Once

func (LocalOnce) Bad() {} // ERROR "Bad passes Lock by value: testdata.LocalOnce contains sync.Mutex"

// False negative:
// LocalMutex doesn't have a Lock method.
// Nevertheless, it is probably a bad idea to pass it by value.
type LocalMutex sync.Mutex

func (LocalMutex) Bad() {} // WANTED: An error here :(
