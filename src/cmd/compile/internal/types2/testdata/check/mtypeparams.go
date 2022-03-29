// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// If types2.Config.AcceptMethodTypeParams is set,
// the type checker accepts methods that have their
// own type parameter list.

package p

type S struct{}

func (S) m[T any](v T) {}

// TODO(gri) Once we collect interface method type parameters
//           in the parser, we can enable these tests again.
/*
type I interface {
   m[T any](v T)
}

type J interface {
   m[T any](v T)
}

var _ I = S{}
var _ I = J(nil)

type C interface{ n() }

type Sc struct{}

func (Sc) m[T C](v T)

type Ic interface {
   m[T C](v T)
}

type Jc interface {
   m[T C](v T)
}

var _ Ic = Sc{}
var _ Ic = Jc(nil)

// TODO(gri) These should fail because the constraints don't match.
var _ I = Sc{}
var _ I = Jc(nil)

var _ Ic = S{}
var _ Ic = J(nil)
*/