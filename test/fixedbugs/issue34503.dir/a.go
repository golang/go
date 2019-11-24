// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

import "unsafe"

type HookFunc func(x uint64)

var HookV unsafe.Pointer

func Hook(x uint64) {
	(*(*HookFunc)(HookV))(x)
}
