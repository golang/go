// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

var pl int

type NoitfStruct struct {
	F int
	G int
}

//go:nointerface
func (t *NoitfStruct) NoInterfaceMethod() {}
