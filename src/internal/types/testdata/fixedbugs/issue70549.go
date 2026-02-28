// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import "math"

var (
	_ = math.Sqrt
	_ = math.SQrt /* ERROR "undefined: math.SQrt (but have Sqrt)" */
	_ = math.sqrt /* ERROR "name sqrt not exported by package math" */
	_ = math.Foo  /* ERROR "undefined: math.Foo" */
	_ = math.foo  /* ERROR "undefined: math.foo" */
)
