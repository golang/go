// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

func Marshal[foobar any]() {
	_ = NewEncoder[foobar]()
}

func NewEncoder[foobar any]() *Encoder[foobar] {
	return nil
}

type Encoder[foobar any] struct {
}

func (e *Encoder[foobar]) EncodeToken(t Token[foobar]) {

}

type Token[foobar any] any
