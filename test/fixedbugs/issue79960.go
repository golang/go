// compile

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f[T []byte | []rune]() {
	_ = T("")
}

var _ = f[[]rune]
var _ = f[[]byte]
