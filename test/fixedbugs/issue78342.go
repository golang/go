// compile

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type T[P any] struct{ _, _, _, _, _, _, _, _ P }

type U T[T[T[T[T[T[T[T[T[T[T[T[T[T[T[T[T[T[T[T[T[T[T[T[T[T[T[T[T[T[T[T[struct{}]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]

var _ U
