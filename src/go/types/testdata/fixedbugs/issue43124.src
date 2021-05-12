// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

var _ = int(0 /* ERROR invalid use of \.\.\. in conversion to int */ ...)

// test case from issue

type M []string

var (
	x = []string{"a", "b"}
	_ = M(x /* ERROR invalid use of \.\.\. in conversion to M */ ...)
)
