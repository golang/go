// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

// The parser does not accept type parameters for interface methods.
// In the past, type checking the code below led to a crash (#50427).

type T interface{ m[ /* ERROR "must have no type parameters" */ P any]() }

func _(t T) {
	var _ interface{ m[ /* ERROR "must have no type parameters" */ P any](); n() } = t /* ERROR "does not implement" */
}

// Type parameters on concrete methods are permitted as of Go 1.27.

type S struct{}

func (S) m[P any]() {}

func _(s S) {
	var _ interface{ m[ /* ERROR "must have no type parameters" */ P any](); n() } = s /* ERROR "does not implement" */

}
