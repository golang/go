// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type (
	A[P any]               [10]P
	S[P any]               struct{ f P }
	P[P any]               *P
	M[K comparable, V any] map[K]V
)
