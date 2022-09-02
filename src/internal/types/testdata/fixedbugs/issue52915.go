// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import "unsafe"

type T[P any] struct {
	T /* ERROR illegal cycle */ [P]
}

func _[P any]() {
	_ = unsafe.Sizeof(T[int]{})
	_ = unsafe.Sizeof(struct{ T[int] }{})

	_ = unsafe.Sizeof(T[P]{})
	_ = unsafe.Sizeof(struct{ T[P] }{})
}

// TODO(gri) This is a follow-on error due to T[int] being invalid.
//           We should try to avoid it.
const _ = unsafe /* ERROR not constant */ .Sizeof(T[int]{})
