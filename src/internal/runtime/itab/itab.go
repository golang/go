// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package itab

import "internal/abi"

// layout of Itab known to compilers
// allocated in non-garbage-collected memory
// Needs to be in sync with
// ../cmd/compile/internal/reflectdata/reflect.go:/^func.WritePluginTable.
type Itab struct {
	Inter *abi.InterfaceType
	Type  *abi.Type
	Hash  uint32 // copy of _type.hash. Used for type switches.
	_     [4]byte
	Fun   [1]uintptr // variable sized. fun[0]==0 means _type does not implement inter.
}
