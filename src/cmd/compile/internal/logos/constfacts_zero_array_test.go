// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"os"
	"path/filepath"
	"testing"

	"cmd/compile/internal/logos/facts"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
	"cmd/internal/goobj"
)

func TestGoMutationFactsZeroArrays(t *testing.T) {
	const source = `package zeroarrays

type Value struct {
	Marker [0]func()
	N int
}

type Live struct { Marker [1]func() }

type Receiver struct { Next *Receiver }

var callValue func(Value)
var callLive func(Live)

func ReturnZero(x Value) any { return x }
func ReturnLive(x Live) any { return x }
func ForwardZero(x Value) { callValue(x) }
func ForwardLive(x Live) { callLive(x) }
func ReturnZeroPointers(x [0]*int) any { return x }
func ReturnLivePointers(x [1]*int) any { return x }
func ReturnPointerToZero(x *[0]func()) any { return x }
func ReturnSliceOfZero(x [][0]func()) any { return x }
func ReturnLiveSibling(x struct { Marker [0]func(); P *int }) any { return x }

func (v Value) Unknown(x Value) Value
func (v Value) Invoke(x Value) Value { return v.Unknown(x) }
func (r *Receiver) Return(x Value) any { return x }
func (r *Receiver) Store(x Value) { r.Next = r; _ = x }

type GenericZero[T any] struct { Items [0]T }
func (g GenericZero[T]) Return(x GenericZero[T]) any { return x }

type GenericLive[T any] struct { Items [1]T }
func (g GenericLive[T]) Return(x GenericLive[T]) any { return x }
`
	path := filepath.Join(t.TempDir(), "facts.go")
	if err := os.WriteFile(path, []byte(source), 0o644); err != nil {
		t.Fatal(err)
	}
	file, err := syntax.ParseFile(path, func(err error) { t.Fatalf("parse: %v", err) }, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	info := &types2.Info{
		Types:      make(map[syntax.Expr]types2.TypeAndValue),
		Defs:       make(map[*syntax.Name]types2.Object),
		Uses:       make(map[*syntax.Name]types2.Object),
		Selections: make(map[*syntax.SelectorExpr]*types2.Selection),
	}
	conf := types2.Config{}
	pkg, err := conf.Check("example.com/zeroarrays", []*syntax.File{file}, info)
	if err != nil {
		t.Fatalf("check Go source: %v", err)
	}
	emitted := facts.EmitGoMutations(pkg, info, []*syntax.File{file}, nil)
	if emitted == nil {
		t.Fatal("EmitGoMutations returned no facts")
	}
	back, err := facts.Decode(facts.Encode(goobj.FingerprintType{1, 2, 3, 4, 5, 6, 7, 8}, emitted))
	if err != nil {
		t.Fatalf("mutation fact round trip: %v", err)
	}
	byName := make(map[string]facts.FnFact)
	for _, fn := range back.Fns {
		if _, duplicate := byName[fn.Object]; duplicate {
			t.Fatalf("duplicate function fact %q", fn.Object)
		}
		byName[fn.Object] = fn
	}
	want := []struct {
		object   string
		mask     string
		receiver bool
	}{
		{"ReturnZero", "0", false},
		{"ReturnLive", "1", false},
		{"ForwardZero", "0", false},
		{"ForwardLive", "1", false},
		{"ReturnZeroPointers", "0", false},
		{"ReturnLivePointers", "1", false},
		{"ReturnPointerToZero", "1", false},
		{"ReturnSliceOfZero", "1", false},
		{"ReturnLiveSibling", "1", false},
		{"Value.Invoke", "0", true},
		{"(*Receiver).Return", "0", false},
		{"(*Receiver).Store", "0", true},
		{"GenericZero.Return", "0", false},
		{"GenericLive.Return", "1", false},
	}
	if len(byName) != len(want) {
		t.Fatalf("function fact count = %d, want %d: %v", len(byName), len(want), byName)
	}
	for _, tc := range want {
		t.Run(tc.object, func(t *testing.T) {
			fn, ok := byName[tc.object]
			if !ok {
				t.Fatalf("missing mutation fact %q", tc.object)
			}
			if fn.ParamMutated != tc.mask || fn.ReceiverMutated != tc.receiver {
				t.Fatalf("%s: ParamMutated=%q ReceiverMutated=%v, want %q %v", tc.object, fn.ParamMutated, fn.ReceiverMutated, tc.mask, tc.receiver)
			}
		})
	}
}
