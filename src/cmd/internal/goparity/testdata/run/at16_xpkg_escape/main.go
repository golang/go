// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"

	"example.com/parity/at16_xpkg_escape/a"
)

func clobber(depth int) {
	var buf [4096]byte
	for i := range buf {
		buf[i] = byte(depth + i)
	}
	if depth > 0 {
		clobber(depth - 1)
	}
	for i := range buf {
		if buf[i] != byte(depth+i) {
			panic("clobber lost")
		}
	}
}

func poison(depth int) {
	var buf [4096]byte
	for i := range buf {
		buf[i] = 0xFF
	}
	if depth > 0 {
		poison(depth - 1)
	}
}

func fill(s []byte) {
	for i := range s {
		s[i] = byte(170 - i%251)
	}
}

func check(name string, s []byte, bad *bool) {
	for i := range s {
		if s[i] != byte(170-i%251) {
			fmt.Printf("LAW %s byte %d got %d\n", name, i, s[i])
			*bad = true
			return
		}
	}
	fmt.Printf("%s ok\n", name)
}

//go:noinline
func stageStore() {
	var x [64]byte
	fill(x[:])
	a.StoreParam(x[:])
}

//go:noinline
func stageBranchF() {
	var x [64]byte
	fill(x[:])
	a.LeakBranch(x[:], false)
}

//go:noinline
func stageBranchT() {
	var x [64]byte
	fill(x[:])
	a.LeakBranch(x[:], true)
}

var gi *int

//go:noinline
func stageMut() {
	var v int
	v = 0
	gi = &v
	a.SetV(gi, 42)
}

func main() {
	bad := false
	stageBranchF()
	clobber(64)
	poison(64)
	if a.SavedBranch != nil {
		fmt.Printf("LAW branch-false retained %d bytes\n", len(a.SavedBranch))
		bad = true
	} else {
		fmt.Printf("branch-false ok\n")
	}
	stageStore()
	clobber(64)
	poison(200)
	check("store", a.Saved, &bad)
	stageBranchT()
	clobber(64)
	poison(200)
	check("branch", a.SavedBranch, &bad)
	stageMut()
	clobber(64)
	poison(200)
	if gi == nil || *gi != 42 {
		fmt.Printf("LAW mutstore\n")
		bad = true
	} else {
		fmt.Printf("mutstore ok\n")
	}
	p := &a.Pointee{}
	a.StoreInto(p, 42)
	clobber(64)
	poison(64)
	if p.V != 42 {
		fmt.Printf("LAW pointee got %d\n", p.V)
		bad = true
	} else {
		fmt.Printf("pointee ok\n")
	}
	e := a.MakeE(7)
	if e.M() != 7 || e.N() != 8 {
		fmt.Printf("LAW embedded got %d %d\n", e.M(), e.N())
		bad = true
	} else {
		fmt.Printf("embedded ok\n")
	}
	em := a.MakeEmpty(7)
	if em.(a.I).M() != 7 {
		fmt.Printf("LAW empty got %d\n", em.(a.I).M())
		bad = true
	} else {
		fmt.Printf("empty ok\n")
	}
	u := a.MakeU(7)
	if u.M() != 7 {
		fmt.Printf("LAW unexported got %d\n", u.M())
		bad = true
	} else {
		fmt.Printf("unexported ok\n")
	}
	g := a.MakeGI(9)
	if g.Get() != 9 {
		fmt.Printf("LAW generic got %d\n", g.Get())
		bad = true
	} else {
		fmt.Printf("generic ok\n")
	}
	if bad {
		os.Exit(1)
	}
}
