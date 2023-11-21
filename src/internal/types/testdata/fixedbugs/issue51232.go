// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type RC[RG any] interface {
	~[]RG
}

type Fn[RCT RC[RG], RG any] func(RCT)

type F[RCT RC[RG], RG any] interface {
	Fn() Fn /* ERROR "not enough type arguments for type Fn: have 1, want 2" */ [RCT]
}

type concreteF[RCT RC[RG], RG any] struct {
	makeFn func() Fn /* ERROR "not enough type arguments for type Fn: have 1, want 2" */ [RCT]
}

func (c *concreteF[RCT, RG]) Fn() Fn /* ERROR "not enough type arguments for type Fn: have 1, want 2" */ [RCT] {
	return c.makeFn()
}

func NewConcrete[RCT RC[RG], RG any](Rc RCT) F /* ERROR "not enough type arguments for type F: have 1, want 2" */ [RCT] {
	// TODO(rfindley): eliminate the duplicate error below.
	return & /* ERRORx `cannot use .* as F\[RCT\]` */ concreteF /* ERROR "not enough type arguments for type concreteF: have 1, want 2" */ [RCT]{
		makeFn: nil,
	}
}
