// compile

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 19084: SSA doesn't handle CONVNOP STRUCTLIT

package p

type T struct {
	a, b, c, d, e, f, g, h int // big, not SSA-able
}

func f() {
	_ = T(T{})
}
