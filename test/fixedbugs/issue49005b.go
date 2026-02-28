// errorcheck

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type T interface{ M() }

func F() T

var _ = F().(*X) // ERROR "impossible type assertion:( F\(\).\(\*X\))?\n\t\*X does not implement T \(missing M method\)"

type X struct{}
