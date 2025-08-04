// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package issue63285

type A[_ B[any]] struct{}

type B[_ any] interface {
	f() A[B[any]]
}
