// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"log"
	"path/filepath"
	"plugin"
	"strings"

	"common"
)

func init() {
	common.X *= 5
}

// testUnnamed tests that two plugins built with .go files passed on
// the command line do not have overlapping symbols. That is,
// unnamed1.so/FuncInt and unnamed2.so/FuncInt should be distinct functions.
func testUnnamed() {
	p, err := plugin.Open("unnamed1.so")
	if err != nil {
		log.Fatalf(`plugin.Open("unnamed1.so"): %v`, err)
	}
	fn, err := p.Lookup("FuncInt")
	if err != nil {
		log.Fatalf(`unnamed1.so: Lookup("FuncInt") failed: %v`, err)
	}
	if got, want := fn.(func() int)(), 1; got != want {
		log.Fatalf("unnamed1.so: FuncInt()=%d, want %d", got, want)
	}

	p, err = plugin.Open("unnamed2.so")
	if err != nil {
		log.Fatalf(`plugin.Open("unnamed2.so"): %v`, err)
	}
	fn, err = p.Lookup("FuncInt")
	if err != nil {
		log.Fatalf(`unnamed2.so: Lookup("FuncInt") failed: %v`, err)
	}
	if got, want := fn.(func() int)(), 2; got != want {
		log.Fatalf("unnamed2.so: FuncInt()=%d, want %d", got, want)
	}
}

func main() {
	if got, want := common.X, 3*5; got != want {
		log.Fatalf("before plugin load common.X=%d, want %d", got, want)
	}

	p, err := plugin.Open("plugin1.so")
	if err != nil {
		log.Fatalf("plugin.Open failed: %v", err)
	}

	const wantX = 3 * 5 * 7
	if got := common.X; got != wantX {
		log.Fatalf("after plugin load common.X=%d, want %d", got, wantX)
	}

	seven, err := p.Lookup("Seven")
	if err != nil {
		log.Fatalf(`Lookup("Seven") failed: %v`, err)
	}
	if got, want := *seven.(*int), 7; got != want {
		log.Fatalf("plugin1.Seven=%d, want %d", got, want)
	}

	readFunc, err := p.Lookup("ReadCommonX")
	if err != nil {
		log.Fatalf(`plugin1.Lookup("ReadCommonX") failed: %v`, err)
	}
	if got := readFunc.(func() int)(); got != wantX {
		log.Fatalf("plugin1.ReadCommonX()=%d, want %d", got, wantX)
	}

	// sub/plugin1.so is a different plugin with the same name as
	// the already loaded plugin. It also depends on common. Test
	// that we can load the different plugin, it is actually
	// different, and that it sees the same common package.
	subpPath, err := filepath.Abs("sub/plugin1.so")
	if err != nil {
		log.Fatalf("filepath.Abs(%q) failed: %v", subpPath, err)
	}
	subp, err := plugin.Open(subpPath)
	if err != nil {
		log.Fatalf("plugin.Open(%q) failed: %v", subpPath, err)
	}

	funcVar, err := subp.Lookup("FuncVar")
	if err != nil {
		log.Fatalf(`sub/plugin1.Lookup("FuncVar") failed: %v`, err)
	}
	called := false
	*funcVar.(*func()) = func() {
		called = true
	}

	readFunc, err = subp.Lookup("ReadCommonX")
	if err != nil {
		log.Fatalf(`sub/plugin1.Lookup("ReadCommonX") failed: %v`, err)
	}
	if got := readFunc.(func() int)(); got != wantX {
		log.Fatalf("sub/plugin1.ReadCommonX()=%d, want %d", got, wantX)
	}
	if !called {
		log.Fatal("calling ReadCommonX did not call FuncVar")
	}

	subf, err := subp.Lookup("F")
	if err != nil {
		log.Fatalf(`sub/plugin1.Lookup("F") failed: %v`, err)
	}
	if gotf := subf.(func() int)(); gotf != 17 {
		log.Fatalf(`sub/plugin1.F()=%d, want 17`, gotf)
	}
	f, err := p.Lookup("F")
	if err != nil {
		log.Fatalf(`plugin1.Lookup("F") failed: %v`, err)
	}
	if gotf := f.(func() int)(); gotf != 3 {
		log.Fatalf(`plugin1.F()=%d, want 17`, gotf)
	}

	// plugin2 has no exported symbols, only an init function.
	if _, err := plugin.Open("plugin2.so"); err != nil {
		log.Fatalf("plugin.Open failed: %v", err)
	}
	if got, want := common.X, 2; got != want {
		log.Fatalf("after loading plugin2, common.X=%d, want %d", got, want)
	}

	_, err = plugin.Open("plugin-mismatch.so")
	if err == nil {
		log.Fatal(`plugin.Open("plugin-mismatch.so"): should have failed`)
	}
	if s := err.Error(); !strings.Contains(s, "different version") {
		log.Fatalf(`plugin.Open("plugin-mismatch.so"): error does not mention "different version": %v`, s)
	}

	testUnnamed()

	fmt.Println("PASS")
}
