// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type S struct {
	x int
}

func V() interface{} {
	return S{0}
}
