// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Go definitions of internal structures. Master is runtime.h

package runtime

import "unsafe"

const (
	Windows = 1
)

// const ( Structrnd = sizeof(uintptr) )

type lock struct {
	key   uint32
	event unsafe.Pointer
}

type note lock
