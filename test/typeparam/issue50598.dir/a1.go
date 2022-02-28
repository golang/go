// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a1

import "a0"

func New() int {
	return a0.IntBuilder{}.New()
}
