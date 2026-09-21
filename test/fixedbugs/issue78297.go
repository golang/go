// compile

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func G[U any]() (u U) { return }

//go:noinline
func H[U any]() (u U) { return }

func F[T ~*[1]byte]() {
	_ = G[T]()[:]
	_ = H[T]()[:]
}

var _ = F[*[1]byte]
