// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

func F() { // ERROR "can inline F"
	g(0) // ERROR "inlining call to g\[go.shape.int\]"
}

func g[T any](_ T) {} // ERROR "can inline g\[int\]" "can inline g\[go.shape.int\]" "inlining call to g\[go.shape.int\]"
