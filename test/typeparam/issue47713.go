// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"encoding"
	"fmt"
)

type Seralizable interface {
	encoding.BinaryMarshaler
	encoding.BinaryUnmarshaler
}

type SerDeString string

func (s *SerDeString) UnmarshalBinary(in []byte) error {
	*s = SerDeString(in)
	return nil
}

func (s SerDeString) MarshalBinary() ([]byte, error) {
	return []byte(s), nil
}


type GenericSerializable[T Seralizable] struct {
	Key string
	Value T
}

func (g GenericSerializable[T]) Send() {
	out, err := g.Value.MarshalBinary()
	if err != nil {
		panic("bad")
	}
	var newval SerDeString
	newval.UnmarshalBinary(out)
	fmt.Printf("Sent %s\n", newval)
}

func main() {
	val := SerDeString("asdf")
	x := GenericSerializable[*SerDeString]{
		Value: &val,
	}
	x.Send()
}
