// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The exvar package provides a standardized interface to public variables,
// such as operation counters in servers.
package exvar

import (
	"fmt";
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

type exVars struct {
	vars map[string] exVar;
	// TODO(dsymonds): docstrings
}

// Singleton worker goroutine.
// Functions needing access to the global state have to pass a closure to the
// worker channel, which is read by a single workerFunc running in a goroutine.
// Nil values are silently ignored, so you can send nil to the worker channel
// after the closure if you want to block until your work is done. This risks
// blocking you, though. The workSync function wraps this as a convenience.

type workFunction func(*exVars);

// The main worker function that runs in a goroutine.
// It never ends in normal operation.
func startWorkerFunc() <-chan workFunction {
	ch := make(chan workFunction);

	state := &exVars{ make(map[string] exVar) };

	go func() {
		for f := range ch {
			if f != nil {
				f(state)
			}
		}
	}();
	return ch
}

var worker = startWorkerFunc();

// workSync will enqueue the given workFunction and wait for it to finish.
func workSync(f workFunction) {
	worker <- f;
	worker <- nil  // will only be sent after f() completes.
}

// getOrInitIntVar either gets or initializes an intVar called name.
func (state *exVars) getOrInitIntVar(name string) *intVar {
	if v, ok := state.vars[name]; ok {
		// Existing var
		if iv, ok := v.(*intVar); ok {
			return iv
		}
		// Type mismatch.
		return state.getOrInitIntVar(mismatchedInt)
	}
	// New var
	iv := new(intVar);
	state.vars[name] = iv;
	return iv
}

// getOrInitMapVar either gets or initializes a mapVar called name.
func (state *exVars) getOrInitMapVar(name string) *mapVar {
	if v, ok := state.vars[name]; ok {
		// Existing var
		if mv, ok := v.(*mapVar); ok {
			return mv
		}
		// Type mismatch.
		return state.getOrInitMapVar(mismatchedMap)
	}
	// New var
	var m mapVar = make(map[string] int);
	state.vars[name] = &m;
	return &m
}

// IncrementInt adds inc to the integer-valued var called name.
func IncrementInt(name string, inc int) {
	workSync(func(state *exVars) {
		*state.getOrInitIntVar(name) += inc
	})
}

// IncrementMapInt adds inc to the keyed value in the map-valued var called name.
func IncrementMapInt(name string, key string, inc int) {
	workSync(func(state *exVars) {
		mv := state.getOrInitMapVar(name);
		// TODO(dsymonds): Change this to just mv[key] when bug143 is fixed.
		if v, ok := (*mv)[key]; ok {
			mv[key] += inc
		} else {
			mv[key] = inc
		}
	})
}

// SetInt sets the integer-valued var called name to value.
func SetInt(name string, value int) {
	workSync(func(state *exVars) {
		*state.getOrInitIntVar(name) = value
	})
}

// SetMapInt sets the keyed value in the map-valued var called name.
func SetMapInt(name string, key string, value int) {
	workSync(func(state *exVars) {
		state.getOrInitMapVar(name)[key] = value
	})
}

// GetInt retrieves an integer-valued var called name.
func GetInt(name string) int {
	var i int;
	workSync(func(state *exVars) {
		i = *state.getOrInitIntVar(name)
	});
	return i
}

// GetMapInt retrieves the keyed value for a map-valued var called name.
func GetMapInt(name string, key string) int {
	var i int;
	var ok bool;
	workSync(func(state *exVars) {
		// TODO(dsymonds): Change this to just getOrInitMapVar(name)[key] when
		// bug143 is fixed.
		i, ok = (*state.getOrInitMapVar(name))[key];
	});
	return i
}

// String produces a string of all the vars in textual format.
func String() string {
	s := "";
	workSync(func(state *exVars) {
		for name, value := range state.vars {
			s += fmt.Sprintln(name, value)
		}
	});
	return s
}
