// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"debug/goobj"
	"reflect"
	"strings"
	"testing"
)

// Each test case is an object file, generated from a corresponding .s file.
// The symbols in the object file with a dead_ prefix are the ones that
// should be removed from the program.
var deadTests = []string{
	"testdata/dead.6",
}

func TestDead(t *testing.T) {
	for _, obj := range deadTests {
		p := Prog{GOOS: "darwin", GOARCH: "amd64", StartSym: "start"}
		p.omitRuntime = true
		p.Error = func(s string) { t.Error(s) }
		p.init()
		p.scan(obj)
		if p.NumError > 0 {
			continue // already reported
		}
		origSyms := copyMap(p.Syms)
		origMissing := copyMap(p.Missing)
		origSymOrder := copySlice(p.SymOrder)
		origPkgSyms := copySlice(p.Packages["main"].Syms)
		p.dead()
		checkDeadMap(t, obj, "p.Syms", origSyms, p.Syms)
		checkDeadMap(t, obj, "p.Missing", origMissing, p.Missing)
		checkDeadSlice(t, obj, "p.SymOrder", origSymOrder, p.SymOrder)
		checkDeadSlice(t, obj, `p.Packages["main"].Syms`, origPkgSyms, p.Packages["main"].Syms)
	}
}

func copyMap(m interface{}) interface{} {
	v := reflect.ValueOf(m)
	out := reflect.MakeMap(v.Type())
	for _, key := range v.MapKeys() {
		out.SetMapIndex(key, v.MapIndex(key))
	}
	return out.Interface()
}

func checkDeadMap(t *testing.T, obj, name string, old, new interface{}) {
	vold := reflect.ValueOf(old)
	vnew := reflect.ValueOf(new)
	for _, vid := range vold.MapKeys() {
		id := vid.Interface().(goobj.SymID)
		if strings.HasPrefix(id.Name, "dead_") {
			if vnew.MapIndex(vid).IsValid() {
				t.Errorf("%s: %s contains unnecessary symbol %s", obj, name, id)
			}
		} else {
			if !vnew.MapIndex(vid).IsValid() {
				t.Errorf("%s: %s is missing symbol %s", obj, name, id)
			}
		}
	}
	for _, vid := range vnew.MapKeys() {
		id := vid.Interface().(goobj.SymID)
		if !vold.MapIndex(vid).IsValid() {
			t.Errorf("%s: %s contains unexpected symbol %s", obj, name, id)
		}
	}
}

func copySlice(x []*Sym) (out []*Sym) {
	return append(out, x...)
}

func checkDeadSlice(t *testing.T, obj, name string, old, new []*Sym) {
	for i, s := range old {
		if strings.HasPrefix(s.Name, "dead_") {
			continue
		}
		if len(new) == 0 {
			t.Errorf("%s: %s is missing symbol %s\nhave%v\nwant%v", obj, name, s, new, old[i:])
			return
		}
		if new[0].SymID != s.SymID {
			t.Errorf("%s: %s is incorrect: have %s, want %s\nhave%v\nwant%v", obj, name, new[0].SymID, s.SymID, new, old[i:])
			return
		}
		new = new[1:]
	}
	if len(new) > 0 {
		t.Errorf("%s: %s has unexpected symbols: %v", new)
	}
}
