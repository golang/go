// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import "time"

// This type declaration must not cause problems with
// the type validity checker.

type S[T any] struct {
	a T
	b time.Time
}

var _ S[time.Time]
