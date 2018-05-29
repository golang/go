// errorcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests that all types and functions are type-checked before any constant
// declaration is. Issue #24755.
package p

type I interface{ F() }
type T struct{}

const _ = I(T{}) // ERROR "const initializer I\(T literal\) is not a constant"

func (T) F() {}
