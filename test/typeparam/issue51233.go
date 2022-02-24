// compile -G=3

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package p

type RC[RG any] interface {
	~[]RG
}
type Fn[RCT RC[RG], RG any] func(RCT)
type FFn[RCT RC[RG], RG any] func() Fn[RCT]
type F[RCT RC[RG], RG any] interface {
	Fn() Fn[RCT]
}
type concreteF[RCT RC[RG], RG any] struct {
	makeFn FFn[RCT]
}

func (c *concreteF[RCT, RG]) Fn() Fn[RCT] {
	return c.makeFn()
}
