// errorcheck

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import "io"

type T1 interface {
	io.Reader
}

type T2 struct {
	io.SectionReader
}

type T3 struct { // ERROR "invalid recursive type: T3 refers to itself"
	T1
	T2
	parent T3
}
