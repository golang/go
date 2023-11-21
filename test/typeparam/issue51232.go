// errorcheck

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type RC[RG any] interface {
	~[]RG
}

type Fn[RCT RC[RG], RG any] func(RCT)

type F[RCT RC[RG], RG any] interface {
	Fn() Fn[RCT] // ERROR "not enough type arguments for type Fn: have 1, want 2"
}

type concreteF[RCT RC[RG], RG any] struct {
	makeFn func() Fn[RCT] // ERROR "not enough type arguments for type Fn: have 1, want 2"
}

func (c *concreteF[RCT, RG]) Fn() Fn[RCT] { // ERROR "not enough type arguments for type Fn: have 1, want 2"
	return c.makeFn()
}

func NewConcrete[RCT RC[RG], RG any](Rc RCT) F[RCT] { // ERROR "not enough type arguments for type F: have 1, want 2"
	return &concreteF[RCT]{ // ERROR "cannot use" "not enough type arguments for type concreteF: have 1, want 2"
		makeFn: nil,
	}
}
