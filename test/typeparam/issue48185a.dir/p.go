// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type MarshalOptions struct {
	Marshalers *Marshalers
}

type Encoder struct {}

type Marshalers = marshalers[MarshalOptions, Encoder]

type marshalers[Options, Coder any] struct{}

func MarshalFuncV1[T any](fn func(T) ([]byte, error)) *Marshalers {
	return &Marshalers{}
}
