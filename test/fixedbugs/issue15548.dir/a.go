// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type I0 interface {
	I1
}

type T struct {
	I1
}

type I1 interface {
	M(*T) // removing * makes crash go away
}
