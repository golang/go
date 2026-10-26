// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"internal/abi"
	"reflect"
)

type M map[int]int

var sink M

// M is used, not merely declared, so that the group type is built early
// enough for its noalg descriptor to be written first.
func useMap() { sink = M{1: 2} }

func main() {
	useMap()
	x := reflect.Zero(reflect.ArrayOf(abi.MapGroupSlots, reflect.TypeFor[int]())).Interface()
	if _, ok := x.([abi.MapGroupSlots]int); !ok {
		panic("array type from reflect.ArrayOf does not match the compiled one")
	}
}
