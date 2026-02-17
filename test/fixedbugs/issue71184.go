// compile

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x

func F[T int32]() {
	_ = G[*[0]T]()[:]
}

func G[T any]() (v T) {
	return
}

var _ = F[int32]
