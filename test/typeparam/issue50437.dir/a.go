// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type MarshalOptions struct {
	*typedArshalers[MarshalOptions]
}

func Marshal(in interface{}) (out []byte, err error) {
	return MarshalOptions{}.Marshal(in)
}

func (mo MarshalOptions) Marshal(in interface{}) (out []byte, err error) {
	err = mo.MarshalNext(in)
	return nil, err
}

func (mo MarshalOptions) MarshalNext(in interface{}) error {
	a := new(arshaler)
	a.marshal = func(MarshalOptions) error { return nil }
	return a.marshal(mo)
}

type arshaler struct {
	marshal func(MarshalOptions) error
}

type typedArshalers[Options any] struct {
	m M
}

func (a *typedArshalers[Options]) lookup(fnc func(Options) error) (func(Options) error, bool) {
	a.m.Load(nil)
	return fnc, false
}

type M struct {}

func (m *M) Load(key any) (value any, ok bool) {
	return
}
