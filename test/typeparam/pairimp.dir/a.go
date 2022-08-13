// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type Pair[F1, F2 any] struct {
	Field1 F1
	Field2 F2
}
