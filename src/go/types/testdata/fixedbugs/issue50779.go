// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type AC interface {
	C
}

type ST []int

type R[S any, P any] struct{}

type SR = R[SS, ST]

type SS interface {
	NSR(any) *SR // ERROR invalid use of type alias SR in recursive type
}

type C interface {
	NSR(any) *SR
}
