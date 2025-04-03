// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import "math"

var (
	_ = math.Sin
	_ = math.SIn /* ERROR "undefined: math.SIn (but have Sin)" */
	_ = math.sin /* ERROR "name sin not exported by package math" */
	_ = math.Foo /* ERROR "undefined: math.Foo" */
	_ = math.foo /* ERROR "undefined: math.foo" */
)
