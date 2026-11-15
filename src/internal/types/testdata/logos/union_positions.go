// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type U interface {
	int | string
}

var u U /* ERROR "interface contains type constraints" */

type Open interface {
	~int | ~string
}

var _ Open // ERROR "interface contains type constraints"

var _ interface /* ERROR "interface contains type constraints" */ {
	int | string
}

var v interface /* ERROR "interface contains type constraints" */ {
	~int | ~string
}
