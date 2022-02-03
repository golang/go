// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package b

import "a"

type Ap1[A, B any] struct {
	opt a.Option[A]
}

type Ap2[A, B any] struct {
	opt a.Option[A]
}
