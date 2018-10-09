// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the copylock checker's
// range statement analysis.

package a

import "sync"

func rangeMutex() {
	var mu sync.Mutex
	var i int

	var s []sync.Mutex
	for range s {
	}
	for i = range s {
	}
	for i := range s {
	}
	for i, _ = range s {
	}
	for i, _ := range s {
	}
	for _, mu = range s { // want "range var mu copies lock: sync.Mutex"
	}
	for _, m := range s { // want "range var m copies lock: sync.Mutex"
	}
	for i, mu = range s { // want "range var mu copies lock: sync.Mutex"
	}
	for i, m := range s { // want "range var m copies lock: sync.Mutex"
	}

	var a [3]sync.Mutex
	for _, m := range a { // want "range var m copies lock: sync.Mutex"
	}

	var m map[sync.Mutex]sync.Mutex
	for k := range m { // want "range var k copies lock: sync.Mutex"
	}
	for mu, _ = range m { // want "range var mu copies lock: sync.Mutex"
	}
	for k, _ := range m { // want "range var k copies lock: sync.Mutex"
	}
	for _, mu = range m { // want "range var mu copies lock: sync.Mutex"
	}
	for _, v := range m { // want "range var v copies lock: sync.Mutex"
	}

	var c chan sync.Mutex
	for range c {
	}
	for mu = range c { // want "range var mu copies lock: sync.Mutex"
	}
	for v := range c { // want "range var v copies lock: sync.Mutex"
	}

	// Test non-idents in range variables
	var t struct {
		i  int
		mu sync.Mutex
	}
	for t.i, t.mu = range s { // want "range var t.mu copies lock: sync.Mutex"
	}
}
