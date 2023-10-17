// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type internal struct {
	f1 string
	f2 float64
}

type S struct {
	F struct {
		I internal
	}
}
