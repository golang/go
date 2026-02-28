// errorcheck

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type S struct{}

func (S) _[_ any]() {} // ERROR "method _ must have no type parameters"

type _ interface {
	m[_ any]() // ERROR "interface method must have no type parameters"
}
