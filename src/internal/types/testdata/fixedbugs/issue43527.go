// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

const L = 10

type (
	_        [L]struct{}
	_        [A /* ERROR undeclared name A for array length */ ]struct{}
	_        [B /* ERROR invalid array length B */ ]struct{}
	_[A any] struct{}

	B int
)
