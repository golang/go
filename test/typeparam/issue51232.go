// errorcheck -G=3

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type RC[RG any] interface {
	~[]RG
}

type Fn[RCT RC[RG], RG any] func(RCT)

type F[RCT RC[RG], RG any] interface {
	Fn() Fn[RCT] // ERROR "got 1 arguments"
}

type concreteF[RCT RC[RG], RG any] struct {
	makeFn func() Fn[RCT] // ERROR "got 1 arguments"
}

func (c *concreteF[RCT, RG]) Fn() Fn[RCT] { // ERROR "got 1 arguments"
	return c.makeFn()
}

func NewConcrete[RCT RC[RG], RG any](Rc RCT) F[RCT] { // ERROR "got 1 arguments"
	return &concreteF[RCT]{ // ERROR "cannot use" "got 1 arguments"
		makeFn: nil,
	}
}
