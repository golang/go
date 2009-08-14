// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package big

import "unsafe"

type Word uintptr

const (
	_S = uintptr(unsafe.Sizeof(Word));  // TODO(gri) should Sizeof return a uintptr?
	_W = _S*8;
	_B = 1<<_W;
	_M = _B-1;
	_W2 = _W/2;
	_B2 = 1<<_W2;
	_M2 = _B2-1;
)
