// -gotypesalias=1

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type d[T any] struct{}
type (
	b d[a]
)

type a = func(c)
type c struct{ a }
