// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

// As of issue #51527, type-type inference has been disabled.

type RC[RG any] interface {
	~[]RG
}

type Fn[RCT RC[RG], RG any] func(RCT)

type FFn[RCT RC[RG], RG any] func() Fn /* ERROR "not enough type arguments for type Fn: have 1, want 2" */ [RCT]

type F[RCT RC[RG], RG any] interface {
	Fn() Fn /* ERROR "not enough type arguments for type Fn: have 1, want 2" */ [RCT]
}

type concreteF[RCT RC[RG], RG any] struct {
	makeFn FFn /* ERROR "not enough type arguments for type FFn: have 1, want 2" */ [RCT]
}

func (c *concreteF[RCT, RG]) Fn() Fn /* ERROR "not enough type arguments for type Fn: have 1, want 2" */ [RCT] {
	return c.makeFn()
}
