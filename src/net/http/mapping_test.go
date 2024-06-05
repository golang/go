// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"cmp"
	"fmt"
	"slices"
	"strconv"
	"testing"
)

func TestMapping(t *testing.T) {
	var m mapping[int, string]
	for i := 0; i < maxSlice; i++ {
		m.add(i, strconv.Itoa(i))
	}
	if m.m != nil {
		t.Fatal("m.m != nil")
	}
	for i := 0; i < maxSlice; i++ {
		g, _ := m.find(i)
		w := strconv.Itoa(i)
		if g != w {
			t.Fatalf("%d: got %s, want %s", i, g, w)
		}
	}
	m.add(4, "4")
	if m.s != nil {
		t.Fatal("m.s != nil")
	}
	if m.m == nil {
		t.Fatal("m.m == nil")
	}
	g, _ := m.find(4)
	if w := "4"; g != w {
		t.Fatalf("got %s, want %s", g, w)
	}
}

func TestMappingEachPair(t *testing.T) {
	var m mapping[int, string]
	var want []entry[int, string]
	for i := 0; i < maxSlice*2; i++ {
		v := strconv.Itoa(i)
		m.add(i, v)
		want = append(want, entry[int, string]{i, v})

	}

	var got []entry[int, string]
	m.eachPair(func(k int, v string) bool {
		got = append(got, entry[int, string]{k, v})
		return true
	})
	slices.SortFunc(got, func(e1, e2 entry[int, string]) int {
		return cmp.Compare(e1.key, e2.key)
	})
	if !slices.Equal(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func BenchmarkFindChild(b *testing.B) {
	key := "articles"
	children := []string{
		"*",
		"cmd.html",
		"code.html",
		"contrib.html",
		"contribute.html",
		"debugging_with_gdb.html",
		"docs.html",
		"effective_go.html",
		"files.log",
		"gccgo_contribute.html",
		"gccgo_install.html",
		"go-logo-black.png",
		"go-logo-blue.png",
		"go-logo-white.png",
		"go1.1.html",
		"go1.2.html",
		"go1.html",
		"go1compat.html",
		"go_faq.html",
		"go_mem.html",
		"go_spec.html",
		"help.html",
		"ie.css",
		"install-source.html",
		"install.html",
		"logo-153x55.png",
		"Makefile",
		"root.html",
		"share.png",
		"sieve.gif",
		"tos.html",
		"articles",
	}
	if len(children) != 32 {
		panic("bad len")
	}
	for _, n := range []int{2, 4, 8, 16, 32} {
		list := children[:n]
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {

			b.Run("rep=linear", func(b *testing.B) {
				var entries []entry[string, any]
				for _, c := range list {
					entries = append(entries, entry[string, any]{c, nil})
				}
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					findChildLinear(key, entries)
				}
			})
			b.Run("rep=map", func(b *testing.B) {
				m := map[string]any{}
				for _, c := range list {
					m[c] = nil
				}
				var x any
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					x = m[key]
				}
				_ = x
			})
			b.Run(fmt.Sprintf("rep=hybrid%d", maxSlice), func(b *testing.B) {
				var h mapping[string, any]
				for _, c := range list {
					h.add(c, nil)
				}
				var x any
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					x, _ = h.find(key)
				}
				_ = x
			})
		})
	}
}

func findChildLinear(key string, entries []entry[string, any]) any {
	for _, e := range entries {
		if key == e.key {
			return e.value
		}
	}
	return nil
}
