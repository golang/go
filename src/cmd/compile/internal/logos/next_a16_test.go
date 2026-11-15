// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import "testing"

func TestNextA16SliceGates(t *testing.T) {
	imp := newTestImporter(t)
	base := "package p\nfunc d(x int) int { return x * 2 }\n"
	cases := []struct {
		name  string
		body  string
		edit  string
		tasks int
	}{
		{name: "sliceC", body: "func g(xs []int) []int {\nvar ys []int\nfor i := 0; i < len(xs); i++ {\nys = append(ys, d(xs[i]))\n}\nreturn ys\n}", edit: "ys := xs.map(x => d(x))"},
		{name: "sliceRangeIdx", body: "func g(xs []int) []int {\nvar ys []int\nfor i := range xs {\nys = append(ys, d(xs[i]))\n}\nreturn ys\n}", edit: "ys := xs.map(x => d(x))"},
		{name: "sliceRangeVal", body: "func g(xs []int) []int {\nvar ys []int\nfor _, x := range xs {\nys = append(ys, d(x))\n}\nreturn ys\n}", edit: "ys := xs.map(v => d(v))"},
		{name: "namedSlice", body: "type S []int\nfunc g(xs S) []int {\nvar ys []int\nfor i := 0; i < len(xs); i++ {\nys = append(ys, d(xs[i]))\n}\nreturn ys\n}", tasks: 1},
		{name: "arrayC", body: "func g(arr [3]int) []int {\nvar ys []int\nfor i := 0; i < len(arr); i++ {\nys = append(ys, d(arr[i]))\n}\nreturn ys\n}", tasks: 1},
		{name: "arrayRange", body: "func g(arr [3]int) []int {\nvar ys []int\nfor _, v := range arr {\nys = append(ys, d(v))\n}\nreturn ys\n}", tasks: 1},
		{name: "stringC", body: "func e(b byte) int { return int(b) }\nfunc g(s string) []int {\nvar out []int\nfor i := 0; i < len(s); i++ {\nout = append(out, e(s[i]))\n}\nreturn out\n}", tasks: 1},
		{name: "stringRange", body: "func g(s string) []int {\nvar out []int\nfor _, c := range s {\nout = append(out, int(c))\n}\nreturn out\n}", tasks: 1},
		{name: "mapRange", body: "func g(m map[string]int) []int {\nvar out []int\nfor _, v := range m {\nout = append(out, d(v))\n}\nreturn out\n}", tasks: 1},
		{name: "ptrArray", body: "func g(p *[3]int) []int {\nvar ys []int\nfor i := 0; i < len(p); i++ {\nys = append(ys, d(p[i]))\n}\nreturn ys\n}", tasks: 1},
		{name: "typeParam", body: "func g[T ~[]int](xs T) []int {\nvar ys []int\nfor _, v := range xs {\nys = append(ys, d(v))\n}\nreturn ys\n}", tasks: 1},
		{name: "chanRange", body: "func g(ch chan int) []int {\nvar out []int\nfor v := range ch {\nout = append(out, d(v))\n}\nreturn out\n}", tasks: 1},
		{name: "intRange", body: "func h(i int) int { return 2 * i }\nfunc g(n int) []int {\nvar out []int\nfor i := range n {\nout = append(out, h(i))\n}\nreturn out\n}", tasks: 1},
		{name: "funcRange", body: "func g(seq func(yield func(int) bool)) []int {\nvar out []int\nfor v := range seq {\nout = append(out, d(v))\n}\nreturn out\n}", tasks: 1},
		{name: "twoVar", body: "func g(xs []int) []int {\nvar ys []int\nfor i, x := range xs {\nys = append(ys, d(x))\n_ = i\n}\nreturn ys\n}", tasks: 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			edits, tasks := scanSnippet(t, imp, base+tc.body+"\n")
			if tc.edit != "" {
				if len(edits) != 1 {
					t.Fatalf("got %d edits", len(edits))
				}
				if len(tasks) != 0 {
					t.Fatalf("got tasks")
				}
				return
			}
			if len(edits) != 0 {
				t.Fatalf("got edits")
			}
			if len(tasks) != tc.tasks {
				t.Fatalf("got %d tasks", len(tasks))
			}
		})
	}
}

func TestNextA16ErrInitValue(t *testing.T) {
	imp := newTestImporter(t)
	call := "package p\nfunc f(s string) (string, error) {\nif s == \"\" {\nreturn \"\", errNew()\n}\nreturn s, nil\n}\nfunc errNew() error { return nil }\n"
	body := "func g(s string) (string, error) {\nif v, err := f(s); err != nil {\nreturn \"\", err\n} else {\nreturn v, nil\n}\n}"
	edits, tasks := scanSnippet(t, imp, call+body+"\n")
	if len(edits) != 0 {
		t.Fatalf("got edits")
	}
	if len(tasks) != 1 {
		t.Fatalf("got %d tasks", len(tasks))
	}
}
