// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import "unsafe"

func _[T any](x T) {
	_ = unsafe.Alignof(x)
	_ = unsafe.Sizeof(x)
}
