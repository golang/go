// Copyright (c) 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package edwards25519

import (
	"testing"
	"testing/quick"
)

func TestScalarAliasing(t *testing.T) {
	checkAliasingOneArg := func(f func(v, x *Scalar) *Scalar, v, x Scalar) bool {
		x1, v1 := x, x

		// Calculate a reference f(x) without aliasing.
		if out := f(&v, &x); out != &v || !isReduced(out) {
			return false
		}

		// Test aliasing the argument and the receiver.
		if out := f(&v1, &v1); out != &v1 || v1 != v || !isReduced(out) {
			return false
		}

		// Ensure the arguments was not modified.
		return x == x1
	}

	checkAliasingTwoArgs := func(f func(v, x, y *Scalar) *Scalar, v, x, y Scalar) bool {
		x1, y1, v1 := x, y, Scalar{}

		// Calculate a reference f(x, y) without aliasing.
		if out := f(&v, &x, &y); out != &v || !isReduced(out) {
			return false
		}

		// Test aliasing the first argument and the receiver.
		v1 = x
		if out := f(&v1, &v1, &y); out != &v1 || v1 != v || !isReduced(out) {
			return false
		}
		// Test aliasing the second argument and the receiver.
		v1 = y
		if out := f(&v1, &x, &v1); out != &v1 || v1 != v || !isReduced(out) {
			return false
		}

		// Calculate a reference f(x, x) without aliasing.
		if out := f(&v, &x, &x); out != &v || !isReduced(out) {
			return false
		}

		// Test aliasing the first argument and the receiver.
		v1 = x
		if out := f(&v1, &v1, &x); out != &v1 || v1 != v || !isReduced(out) {
			return false
		}
		// Test aliasing the second argument and the receiver.
		v1 = x
		if out := f(&v1, &x, &v1); out != &v1 || v1 != v || !isReduced(out) {
			return false
		}
		// Test aliasing both arguments and the receiver.
		v1 = x
		if out := f(&v1, &v1, &v1); out != &v1 || v1 != v || !isReduced(out) {
			return false
		}

		// Ensure the arguments were not modified.
		return x == x1 && y == y1
	}

	for name, f := range map[string]interface{}{
		"Negate": func(v, x Scalar) bool {
			return checkAliasingOneArg((*Scalar).Negate, v, x)
		},
		"Multiply": func(v, x, y Scalar) bool {
			return checkAliasingTwoArgs((*Scalar).Multiply, v, x, y)
		},
		"Add": func(v, x, y Scalar) bool {
			return checkAliasingTwoArgs((*Scalar).Add, v, x, y)
		},
		"Subtract": func(v, x, y Scalar) bool {
			return checkAliasingTwoArgs((*Scalar).Subtract, v, x, y)
		},
	} {
		err := quick.Check(f, &quick.Config{MaxCountScale: 1 << 5})
		if err != nil {
			t.Errorf("%v: %v", name, err)
		}
	}
}
