// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package abi

// The first word of every non-empty interface type contains an *ITab.
// It records the underlying concrete type (Type), the interface type it
// is implementing (Inter), and some ancillary information.
//
// allocated in non-garbage-collected memory
type ITab struct {
	Inter *InterfaceType
	Type  *Type
	Hash  uint32 // copy of Type.Hash. Used for type switches.
	_     [4]byte
	Fun   [1]uintptr // variable sized. fun[0]==0 means Type does not implement Inter.
}
