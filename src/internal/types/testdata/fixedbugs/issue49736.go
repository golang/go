// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import "math/big"

// From go.dev/issue/18419
func _(x *big.Float) {
	x.form /* ERROR "x.form undefined (cannot refer to unexported field form)" */ ()
}

// From go.dev/issue/31053
func _() {
	_ = big.Float{form /* ERROR "cannot refer to unexported field form in struct literal of type big.Float" */ : 0}
}
