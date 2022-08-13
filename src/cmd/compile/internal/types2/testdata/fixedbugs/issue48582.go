// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type N /* ERROR cycle */ interface {
	int | N
}

type A /* ERROR cycle */ interface {
	int | B
}

type B interface {
	int | A
}

type S /* ERROR cycle */ struct {
	I // ERROR interface contains type constraints
}

type I interface {
	int | S
}

type P interface {
	*P // ERROR interface contains type constraints
}
