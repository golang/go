// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the useless-assignment checker.

package testdata

import "math/rand"

type ST struct {
	x int
	l []int
}

func (s *ST) SetX(x int, ch chan int) {
	// Accidental self-assignment; it should be "s.x = x"
	x = x // want "self-assignment of x to x"
	// Another mistake
	s.x = s.x // want "self-assignment of s.x to s.x"

	s.l[0] = s.l[0] // want "self-assignment of s.l.0. to s.l.0."

	// Bail on any potential side effects to avoid false positives
	s.l[num()] = s.l[num()]
	rng := rand.New(rand.NewSource(0))
	s.l[rng.Intn(len(s.l))] = s.l[rng.Intn(len(s.l))]
	s.l[<-ch] = s.l[<-ch]
}

func num() int { return 2 }

func Index() {
	s := []int{1}
	s[0] = s[0] // want "self-assignment"

	var a [5]int
	a[0] = a[0] // want "self-assignment"

	pa := &[2]int{1, 2}
	pa[1] = pa[1] // want "self-assignment"

	var pss *struct { // report self assignment despite nil dereference
		s []int
	}
	pss.s[0] = pss.s[0] // want "self-assignment"

	m := map[int]string{1: "a"}
	m[0] = m[0]     // bail on map self-assignments due to side effects
	m[1] = m[1]     // not modeling what elements must be in the map
	(m[2]) = (m[2]) // even with parens
	type Map map[string]bool
	named := make(Map)
	named["s"] = named["s"] // even on named maps.
	var psm *struct {
		m map[string]int
	}
	psm.m["key"] = psm.m["key"] // handles dereferences
}
