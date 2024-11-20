// compile

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The gofrontend had a bug handling panic of an untyped constant expression.

package issue68734

func F1() {
	panic(1 + 2)
}

func F2() {
	panic("a" + "b")
}
