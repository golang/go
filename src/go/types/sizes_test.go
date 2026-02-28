// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for sizes.

package types_test

import (
	"go/ast"
	"go/token"
	"go/types"
	"internal/testenv"
	"testing"
)

// findStructType typechecks src and returns the first struct type encountered.
func findStructType(t *testing.T, src string) *types.Struct {
	return findStructTypeConfig(t, src, &types.Config{})
}

func findStructTypeConfig(t *testing.T, src string, conf *types.Config) *types.Struct {
	types_ := make(map[ast.Expr]types.TypeAndValue)
	mustTypecheck(src, nil, &types.Info{Types: types_})
	for _, tv := range types_ {
		if ts, ok := tv.Type.(*types.Struct); ok {
			return ts
		}
	}
	t.Fatalf("failed to find a struct type in src:\n%s\n", src)
	return nil
}

// go.dev/issue/16316
func TestMultipleSizeUse(t *testing.T) {
	const src = `
package main

type S struct {
    i int
    b bool
    s string
    n int
}
`
	ts := findStructType(t, src)
	sizes := types.StdSizes{WordSize: 4, MaxAlign: 4}
	if got := sizes.Sizeof(ts); got != 20 {
		t.Errorf("Sizeof(%v) with WordSize 4 = %d want 20", ts, got)
	}
	sizes = types.StdSizes{WordSize: 8, MaxAlign: 8}
	if got := sizes.Sizeof(ts); got != 40 {
		t.Errorf("Sizeof(%v) with WordSize 8 = %d want 40", ts, got)
	}
}

// go.dev/issue/16464
func TestAlignofNaclSlice(t *testing.T) {
	const src = `
package main

var s struct {
	x *int
	y []byte
}
`
	ts := findStructType(t, src)
	sizes := &types.StdSizes{WordSize: 4, MaxAlign: 8}
	var fields []*types.Var
	// Make a copy manually :(
	for i := 0; i < ts.NumFields(); i++ {
		fields = append(fields, ts.Field(i))
	}
	offsets := sizes.Offsetsof(fields)
	if offsets[0] != 0 || offsets[1] != 4 {
		t.Errorf("OffsetsOf(%v) = %v want %v", ts, offsets, []int{0, 4})
	}
}

func TestIssue16902(t *testing.T) {
	const src = `
package a

import "unsafe"

const _ = unsafe.Offsetof(struct{ x int64 }{}.x)
`
	info := types.Info{Types: make(map[ast.Expr]types.TypeAndValue)}
	conf := types.Config{
		// TODO(adonovan): use same FileSet as mustTypecheck.
		Importer: defaultImporter(token.NewFileSet()),
		Sizes:    &types.StdSizes{WordSize: 8, MaxAlign: 8},
	}
	mustTypecheck(src, &conf, &info)
	for _, tv := range info.Types {
		_ = conf.Sizes.Sizeof(tv.Type)
		_ = conf.Sizes.Alignof(tv.Type)
	}
}

// go.dev/issue/53884.
func TestAtomicAlign(t *testing.T) {
	testenv.MustHaveGoBuild(t) // The Go command is needed for the importer to determine the locations of stdlib .a files.

	const src = `
package main

import "sync/atomic"

var s struct {
	x int32
	y atomic.Int64
	z int64
}
`

	want := []int64{0, 8, 16}
	for _, arch := range []string{"386", "amd64"} {
		t.Run(arch, func(t *testing.T) {
			conf := types.Config{
				// TODO(adonovan): use same FileSet as findStructTypeConfig.
				Importer: defaultImporter(token.NewFileSet()),
				Sizes:    types.SizesFor("gc", arch),
			}
			ts := findStructTypeConfig(t, src, &conf)
			var fields []*types.Var
			// Make a copy manually :(
			for i := 0; i < ts.NumFields(); i++ {
				fields = append(fields, ts.Field(i))
			}

			offsets := conf.Sizes.Offsetsof(fields)
			if offsets[0] != want[0] || offsets[1] != want[1] || offsets[2] != want[2] {
				t.Errorf("OffsetsOf(%v) = %v want %v", ts, offsets, want)
			}
		})
	}
}

type gcSizeTest struct {
	name string
	src  string
}

var gcSizesTests = []gcSizeTest{
	{
		"issue60431",
		`
package main

import "unsafe"

// The foo struct size is expected to be rounded up to 16 bytes.
type foo struct {
	a int64
	b bool
}

func main() {
	assert(unsafe.Sizeof(foo{}) == 16)
}`,
	},
	{
		"issue60734",
		`
package main

import (
	"unsafe"
)

// The Data struct size is expected to be rounded up to 16 bytes.
type Data struct {
	Value  uint32   // 4 bytes
	Label  [10]byte // 10 bytes
	Active bool     // 1 byte
	// padded with 1 byte to make it align
}

func main() {
	assert(unsafe.Sizeof(Data{}) == 16)
}
`,
	},
}

func TestGCSizes(t *testing.T) {
	types.DefPredeclaredTestFuncs()
	for _, tc := range gcSizesTests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			conf := types.Config{
				// TODO(adonovan): use same FileSet as mustTypecheck.
				Importer: defaultImporter(token.NewFileSet()),
				Sizes:    types.SizesFor("gc", "amd64"),
			}
			mustTypecheck(tc.src, &conf, nil)
		})
	}
}
