// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

import (
	"reflect"
	"sync"
)

type addressableValue struct{ reflect.Value }

type arshalers[Options, Coder any] struct {
	fncVals  []typedArshaler[Options, Coder]
	fncCache sync.Map // map[reflect.Type]unmarshaler
}
type typedArshaler[Options, Coder any] struct {
	typ reflect.Type
	fnc func(Options, *Coder, addressableValue) error
}

type UnmarshalOptions1 struct {
	// Unmarshalers is a list of type-specific unmarshalers to use.
	Unmarshalers *arshalers[UnmarshalOptions1, Decoder1]
}

type Decoder1 struct {
}

func (a *arshalers[Options, Coder]) lookup(fnc func(Options, *Coder, addressableValue) error, t reflect.Type) func(Options, *Coder, addressableValue) error {
	return fnc
}

func UnmarshalFuncV2[T any](fn func(UnmarshalOptions1, *Decoder1, T) error) *arshalers[UnmarshalOptions1, Decoder1] {
	return &arshalers[UnmarshalOptions1, Decoder1]{}
}
