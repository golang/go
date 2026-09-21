// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f[P any](_ P) {}

func _(x map[func /* ERROR "invalid map key type func(int)" */ (int)]bool) {
	// While functions are not valid map keys, the intended type for P
	// is still unambiguous here. Avoid "cannot use generic function f
	// without instantiation".
	x[f] = true
}

// same thinking applies through a type parameter
func _[P map[func /* ERROR "invalid map key type func(int)" */ (int)]bool](x P) {
	x[f] = true
}
