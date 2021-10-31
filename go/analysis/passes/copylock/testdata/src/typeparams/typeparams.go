// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typeparams

import "sync"

func OkFunc1[Struct ~*struct{ mu sync.Mutex }](s Struct) {
}

func BadFunc1[Struct ~struct{ mu sync.Mutex }](s Struct) { // want `passes lock by value: .*Struct contains ~struct{mu sync.Mutex}`
}

func OkFunc2[MutexPtr *sync.Mutex](m MutexPtr) {
	var x *MutexPtr
	p := x
	var y MutexPtr
	p = &y
	*p = *x

	var mus []MutexPtr

	for _, _ = range mus {
	}
}

func BadFunc2[Mutex sync.Mutex](m Mutex) { // want `passes lock by value: .*Mutex contains sync.Mutex`
	var x *Mutex
	p := x
	var y Mutex
	p = &y
	*p = *x // want `assignment copies lock value to \*p: .*Mutex contains sync.Mutex`

	var mus []Mutex

	for _, _ = range mus {
	}
}

func ApproximationError[Mutex interface {
	~sync.Mutex
	M()
}](m Mutex) { // want `passes lock by value: .*Mutex contains ~sync.Mutex`
}
