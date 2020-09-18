// run

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// short-circuiting interface-to-concrete comparisons
// will not miss panics

package main

import (
	"log"
	"strings"
)

func main() {
	var (
		x interface{}
		p *int
		s []int
		l *interface{}
		r []*int
	)
	tests := []struct {
		name   string
		errStr string
		f      func()
	}{
		{"switch case", "", func() {
			switch x {
			case x.(*int):
			}
		}},
		{"interface conversion", "", func() { _ = x == x.(error) }},
		{"type assertion", "", func() { _ = x == x.(*int) }},
		{"out of bounds", "", func() { _ = x == s[1] }},
		{"nil pointer dereference #1", "", func() { _ = x == *p }},
		{"nil pointer dereference #2", "nil pointer dereference", func() { _ = *l == r[0] }},
	}

	for _, tc := range tests {
		testFuncShouldPanic(tc.name, tc.errStr, tc.f)
	}
}

func testFuncShouldPanic(name, errStr string, f func()) {
	defer func() {
		e := recover()
		if e == nil {
			log.Fatalf("%s: comparison did not panic\n", name)
		}
		if errStr != "" {
			if !strings.Contains(e.(error).Error(), errStr) {
				log.Fatalf("%s: wrong panic message\n", name)
			}
		}
	}()
	f()
}
