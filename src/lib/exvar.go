// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The exvar package provides a standardized interface to public variables,
// such as operation counters in servers.
package exvar

import (
	"fmt";
	"sync";
)

// Global state.
var (
	mutex sync.Mutex;
	intVars = make(map[string] int);
	mapVars = make(map[string] map[string] int);
	// TODO(dsymonds):
	// - string-valued vars
	// - docstrings
	// - dynamic lookup vars (via chan)
)

// Increment adds inc to the var called name.
func Increment(name string, inc int) {
	mutex.Lock();
	defer mutex.Unlock();

	if x, ok := intVars[name]; ok {
		intVars[name] += inc
	} else {
		intVars[name] = inc
	}
}

// Set sets the var called name to value.
func Set(name string, value int) {
	intVars[name] = value
}

// Get retrieves an integer-valued var called name.
func Get(name string) (x int, ok bool) {
	x, ok = intVars[name];
	return
}

// TODO(dsymonds): Functions for map-valued vars.

// String produces a string of all the vars in textual format.
func String() string {
	mutex.Lock();
	defer mutex.Unlock();

	s := "";
	for name, value := range intVars {
		s += fmt.Sprintln(name, value)
	}
	return s
}
