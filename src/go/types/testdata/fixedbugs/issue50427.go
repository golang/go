// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

// The parser no longer parses type parameters for methods.
// In the past, type checking the code below led to a crash (#50427).

type T interface{ m[ /* ERROR "must have no type parameters" */ P any]() }

func _(t T) {
	var _ interface{ m[ /* ERROR "must have no type parameters" */ P any](); n() } = t /* ERROR "does not implement" */
}

type S struct{}

func (S) m[ /* ERROR "must have no type parameters" */ P any]() {}

func _(s S) {
	var _ interface{ m[ /* ERROR "must have no type parameters" */ P any](); n() } = s /* ERROR "does not implement" */

}
