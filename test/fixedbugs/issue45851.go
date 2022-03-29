// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This tickles a stack-allocation bug when the register ABI is enabled.
// The original report was from cue, internal/core/adt/equality.go,
// function equalVertex.

// In the failing case, something bad gets passed to equalTerminal.

package main

import "fmt"

type Kind uint16
type Flag uint16

const (
	allKinds Kind = 1
	TopKind  Kind = (allKinds - 1)
)
type Value interface {
	Kind() Kind
}
type Vertex struct {
	BaseValue Value
	name string
}
func (v *Vertex) Kind() Kind {
	return TopKind
}

func main() {
	vA := &Vertex{name:"vA",}
	vB := &Vertex{name:"vB",}
	vX := &Vertex{name:"vX",}
	vA.BaseValue = vX
	vB.BaseValue = vX
	_ = equalVertex(vA, vB, Flag(1))
}

var foo string

//go:noinline
func (v *Vertex) IsClosedStruct() bool {
	return true
}

func equalVertex(x *Vertex, v Value, flags Flag) bool {
	y, ok := v.(*Vertex)
	if !ok {
		return false
	}
	v, ok1 := x.BaseValue.(Value)
	w, ok2 := y.BaseValue.(Value)
	if !ok1 && !ok2 {
		return true // both are struct or list.
	}
	return equalTerminal(v, w, flags)
}

//go:noinline
func equalTerminal(x Value, y Value, flags Flag) bool {
	foo = fmt.Sprintf("EQclosed %s %s %d\n", x.(*Vertex).name, y.(*Vertex).name, flags)
	return true
}
