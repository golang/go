// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.18
// +build go1.18

package completion

import (
	"testing"

	. "golang.org/x/tools/internal/lsp/regtest"
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
			pos := env.RegexpSearch("main.go", tst.pat)
			pos.Column += len(tst.pat)
			completions := env.Completion("main.go", pos)
			result := compareCompletionResults(tst.want, completions.Items)
			if result != "" {
				t.Errorf("%s: wanted %v", result, tst.want)
				for i, g := range completions.Items {
					t.Errorf("got %d %s %s", i, g.Label, g.Detail)
				}
			}
		}
	})
}
