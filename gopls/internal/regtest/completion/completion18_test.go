// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.18
// +build go1.18

package completion

import (
	"testing"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
	. "golang.org/x/tools/gopls/internal/lsp/regtest"
)

// test generic receivers
func TestGenericReceiver(t *testing.T) {
	const files = `
-- go.mod --
module mod.com

go 1.18
-- main.go --
package main
type SyncMap[K any, V comparable] struct {}
func (s *SyncMap[K,V]) f() {}
type XX[T any] struct {}
type UU[T any] struct {}
func (s SyncMap[XX,string]) g(v UU) {}
`

	tests := []struct {
		pat  string
		want []string
	}{
		{"s .Syn", []string{"SyncMap[K, V]"}},
		{"Map.X", []string{}}, // This is probably wrong, Maybe "XX"?
		{"v U", []string{"UU", "uint", "uint16", "uint32", "uint64", "uint8", "uintptr"}}, // not U[T]
	}
	Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		env.Await(env.DoneWithOpen())
		for _, tst := range tests {
			loc := env.RegexpSearch("main.go", tst.pat)
			loc.Range.Start.Character += uint32(protocol.UTF16Len([]byte(tst.pat)))
			completions := env.Completion(loc)
			result := compareCompletionLabels(tst.want, completions.Items)
			if result != "" {
				t.Errorf("%s: wanted %v", result, tst.want)
				for i, g := range completions.Items {
					t.Errorf("got %d %s %s", i, g.Label, g.Detail)
				}
			}
		}
	})
}
func TestFuzzFunc(t *testing.T) {
	// use the example from the package documentation
	modfile := `
-- go.mod --
module mod.com

go 1.18
`
	part0 := `package foo
import "testing"
func FuzzNone(f *testing.F) {
	f.Add(12) // better not find this f.Add
}
func FuzzHex(f *testing.F) {
	for _, seed := range [][]byte{{}, {0}, {9}, {0xa}, {0xf}, {1, 2, 3, 4}} {
		f.Ad`
	part1 := `d(seed)
	}
	f.F`
	part2 := `uzz(func(t *testing.T, in []byte) {
		enc := hex.EncodeToString(in)
		out, err := hex.DecodeString(enc)
		if err != nil {
		  f.Failed()
		}
		if !bytes.Equal(in, out) {
		  t.Fatalf("%v: round trip: %v, %s", in, out, f.Name())
		}
	})
}
`
	data := modfile + `-- a_test.go --
` + part0 + `
-- b_test.go --
` + part0 + part1 + `
-- c_test.go --
` + part0 + part1 + part2

	tests := []struct {
		file   string
		pat    string
		offset uint32 // UTF16 length from the beginning of pat to what the user just typed
		want   []string
	}{
		{"a_test.go", "f.Ad", 3, []string{"Add"}},
		{"c_test.go", " f.F", 4, []string{"Failed"}},
		{"c_test.go", "f.N", 3, []string{"Name"}},
		{"b_test.go", "f.F", 3, []string{"Fuzz(func(t *testing.T, a []byte)", "Fail", "FailNow",
			"Failed", "Fatal", "Fatalf"}},
	}
	Run(t, data, func(t *testing.T, env *Env) {
		for _, test := range tests {
			env.OpenFile(test.file)
			env.Await(env.DoneWithOpen())
			loc := env.RegexpSearch(test.file, test.pat)
			loc.Range.Start.Character += test.offset // character user just typed? will type?
			completions := env.Completion(loc)
			result := compareCompletionLabels(test.want, completions.Items)
			if result != "" {
				t.Errorf("pat %q %q", test.pat, result)
				for i, it := range completions.Items {
					t.Errorf("%d got %q %q", i, it.Label, it.Detail)
				}
			}
		}
	})
}
