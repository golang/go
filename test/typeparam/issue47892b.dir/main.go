// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "./a"

type S[Idx any] struct {
	A string
	B Idx
}

type O[Idx any] struct {
	A int
	B a.I[Idx]
}
