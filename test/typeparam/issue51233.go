// errorcheck

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package p

// As of issue #51527, type-type inference has been disabled.

type RC[RG any] interface {
	~[]RG
}

type Fn[RCT RC[RG], RG any] func(RCT)

type FFn[RCT RC[RG], RG any] func() Fn[RCT] // ERROR "got 1 arguments"

type F[RCT RC[RG], RG any] interface {
	Fn() Fn[RCT] // ERROR "got 1 arguments"
}

type concreteF[RCT RC[RG], RG any] struct {
	makeFn FFn[RCT] // ERROR "got 1 arguments"
}

func (c *concreteF[RCT, RG]) Fn() Fn[RCT] { // ERROR "got 1 arguments"
	return c.makeFn()
}
