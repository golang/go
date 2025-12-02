// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import "unsafe"

var v any = 42

type T /* ERROR "invalid recursive type" */ struct {
	f [unsafe.Sizeof(v.(T))]int
}
