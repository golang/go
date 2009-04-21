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

// If mismatched names are used (e.g. calling IncrementInt on a mapVar), the
// var name is silently mapped to these. We will consider variables starting
// with reservedPrefix to be reserved by this package, and so we avoid the
// possibility of a user doing IncrementInt("x-mismatched-map", 1).
// TODO(dsymonds): Enforce this.
const (
	reservedPrefix = "x-";
	mismatchedInt = reservedPrefix + "mismatched-int";
	mismatchedMap = reservedPrefix + "mismatched-map";
)

// exVar is an abstract type for all exported variables.
type exVar interface {
	String() string;
}

// intVar is an integer variable, and satisfies the exVar interface.
type intVar int;

func (i intVar) String() string {
	return fmt.Sprint(int(i))
}

// mapVar is a map variable, and satisfies the exVar interface.
type mapVar map[string] int;

func (m mapVar) String() string {
	s := "map:x";  // TODO(dsymonds): the 'x' should be user-specified!
	for k, v := range m {
		s += fmt.Sprintf(" %s:%v", k, v)
	}
	return s
}

// TODO(dsymonds):
// - string-valued vars
// - dynamic lookup vars (via chan?)

// Global state.
var (
	mutex sync.Mutex;
	vars = make(map[string] exVar);
	// TODO(dsymonds): docstrings
)

// getOrInitIntVar either gets or initializes an intVar called name.
// Callers should already be holding the mutex.
func getOrInitIntVar(name string) *intVar {
	if v, ok := vars[name]; ok {
		// Existing var
		if iv, ok := v.(*intVar); ok {
			return iv
		}
		// Type mismatch.
		return getOrInitIntVar(mismatchedInt)
	}
	// New var
	iv := new(intVar);
	vars[name] = iv;
	return iv
}

// getOrInitMapVar either gets or initializes a mapVar called name.
// Callers should already be holding the mutex.
func getOrInitMapVar(name string) *mapVar {
	if v, ok := vars[name]; ok {
		// Existing var
		if mv, ok := v.(*mapVar); ok {
			return mv
		}
		// Type mismatch.
		return getOrInitMapVar(mismatchedMap)
	}
	// New var
	var m mapVar = make(map[string] int);
	vars[name] = &m;
	return &m
}

// IncrementInt adds inc to the integer-valued var called name.
func IncrementInt(name string, inc int) {
	mutex.Lock();
	defer mutex.Unlock();

	*getOrInitIntVar(name) += inc
}

// IncrementMap adds inc to the keyed value in the map-valued var called name.
func IncrementMap(name string, key string, inc int) {
	mutex.Lock();
	defer mutex.Unlock();

	mv := getOrInitMapVar(name);
	// TODO(dsymonds): Change this to just mv[key] when bug143 is fixed.
	if v, ok := (*mv)[key]; ok {
		mv[key] += inc
	} else {
		mv[key] = inc
	}
}

// SetInt sets the integer-valued var called name to value.
func SetInt(name string, value int) {
	mutex.Lock();
	defer mutex.Unlock();

	*getOrInitIntVar(name) = value
}

// SetMap sets the keyed value in the map-valued var called name.
func SetMap(name string, key string, value int) {
	mutex.Lock();
	defer mutex.Unlock();

	getOrInitMapVar(name)[key] = value
}

// GetInt retrieves an integer-valued var called name.
func GetInt(name string) int {
	mutex.Lock();
	defer mutex.Unlock();

	return *getOrInitIntVar(name)
}

// GetMap retrieves the keyed value for a map-valued var called name.
func GetMap(name string, key string) int {
	mutex.Lock();
	defer mutex.Unlock();

	// TODO(dsymonds): Change this to just getOrInitMapVar(name)[key] when
	// bug143 is fixed.
	x, ok := (*getOrInitMapVar(name))[key];
	return x
}

// String produces a string of all the vars in textual format.
func String() string {
	mutex.Lock();
	defer mutex.Unlock();

	s := "";
	for name, value := range vars {
		s += fmt.Sprintln(name, value)
	}
	return s
}
