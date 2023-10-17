// compile

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue #43677: ICE during compilation of dynamic initializers for
// composite blank variables.

package p

func f() *int

var _ = [2]*int{nil, f()}

var _ = struct{ x, y *int }{nil, f()}

var _ interface{} = f()
