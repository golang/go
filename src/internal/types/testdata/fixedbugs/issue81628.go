// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import "unsafe"

var _ unsafe.Pointer = 1 << /* ERROR "constant shift overflow" */ 512
var _ unsafe.Pointer = 1 << /* ERROR "constant shift overflow" */ 1000
var _ unsafe.Pointer = 1 /* ERROR "cannot use" */
var _ unsafe.Pointer = 1 /* ERROR "cannot use" */ << 100
var _ = unsafe.Pointer(1 /* ERROR "cannot convert" */)
var _ = unsafe.Pointer(1 /* ERROR "cannot convert" */ << 100)
var _ = unsafe.Pointer(1 << /* ERROR "constant shift overflow" */ 1000)

var _ unsafe.Pointer = 0x1p1000000000 * 0x1p1000000000 * /* ERROR "constant result is not representable" */ 0x1p1000000000
var _ unsafe.Pointer = 1.0            /* ERROR "cannot use" */
var _ unsafe.Pointer = 0x1p1000000000 /* ERROR "cannot use" */
var _ = unsafe.Pointer(1.0 /* ERROR "cannot convert" */)
var _ = unsafe.Pointer(0x1p1000000000 /* ERROR "cannot convert" */)
var _ = unsafe.Pointer(0x1p1000000000 * 0x1p1000000000 * /* ERROR "constant result is not representable" */ 0x1p1000000000)

var _ unsafe.Pointer = "x" /* ERROR "cannot use" */
var _ = unsafe.Pointer("x" /* ERROR "cannot convert" */)
