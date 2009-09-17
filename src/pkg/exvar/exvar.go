// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The exvar package provides a standardized interface to public variables,
// such as operation counters in servers. It exposes these variables via
// HTTP at /debug/vars in JSON format.
package exvar

import (
	"bytes";
	"fmt";
	"http";
	"log";
	"strconv";
	"sync";
)

// Var is an abstract type for all exported variables.
type Var interface {
	String() string;
}

// Int is a 64-bit integer variable, and satisfies the Var interface.
type Int struct {
	i int64;
	mu sync.Mutex;
}

func (v *Int) String() string {
	return strconv.Itoa64(v.i)
}

func (v *Int) Add(delta int64) {
	v.mu.Lock();
	defer v.mu.Unlock();
	v.i += delta;
}

// Map is a string-to-Var map variable, and satisfies the Var interface.
type Map struct {
	m map[string] Var;
	mu sync.Mutex;
}

// KeyValue represents a single entry in a Map.
type KeyValue struct {
	Key string;
	Value Var;
}

func (v *Map) String() string {
	v.mu.Lock();
	defer v.mu.Unlock();
	b := new(bytes.Buffer);
	fmt.Fprintf(b, "{");
	first := true;
	for key, val := range v.m {
		if !first {
			fmt.Fprintf(b, ", ");
		}
		fmt.Fprintf(b, "\"%s\": %v", key, val.String());
		first = false;
	}
	fmt.Fprintf(b, "}");
	return string(b.Bytes())
}

func (v *Map) Init() *Map {
	v.m = make(map[string] Var);
	return v
}

func (v *Map) Get(key string) Var {
	v.mu.Lock();
	defer v.mu.Unlock();
	if av, ok := v.m[key]; ok {
		return av
	}
	return nil
}

func (v *Map) Set(key string, av Var) {
	v.mu.Lock();
	defer v.mu.Unlock();
	v.m[key] = av;
}

func (v *Map) Add(key string, delta int64) {
	v.mu.Lock();
	defer v.mu.Unlock();
	av, ok := v.m[key];
	if !ok {
		av = new(Int);
		v.m[key] = av;
	}

	// Add to Int; ignore otherwise.
	if iv, ok := av.(*Int); ok {
		iv.Add(delta);
	}
}

// TODO(rsc): Make sure map access in separate thread is safe.
func (v *Map) iterate(c chan<- KeyValue) {
	for k, v := range v.m {
		c <- KeyValue{ k, v };
	}
	close(c);
}

func (v *Map) Iter() <-chan KeyValue {
	c := make(chan KeyValue);
	go v.iterate(c);
	return c
}

// String is a string variable, and satisfies the Var interface.
type String struct {
	s string;
}

func (v *String) String() string {
	return strconv.Quote(v.s)
}

func (v *String) Set(value string) {
	v.s = value;
}

// IntFunc wraps a func() int64 to create a value that satisfies the Var interface.
// The function will be called each time the Var is evaluated.
type IntFunc func() int64;

func (v IntFunc) String() string {
	return strconv.Itoa64(v())
}


// All published variables.
var vars map[string] Var = make(map[string] Var);
var mutex sync.Mutex;

// Publish declares an named exported variable. This should be called from a
// package's init function when it creates its Vars. If the name is already
// registered then this will log.Crash.
func Publish(name string, v Var) {
	mutex.Lock();
	defer mutex.Unlock();
	if _, existing := vars[name]; existing {
		log.Crash("Reuse of exported var name:", name);
	}
	vars[name] = v;
}

// Get retrieves a named exported variable.
func Get(name string) Var {
	if v, ok := vars[name]; ok {
		return v
	}
	return nil
}

// RemoveAll removes all exported variables.
// This is for tests; don't call this on a real server.
func RemoveAll() {
	mutex.Lock();
	defer mutex.Unlock();
	vars = make(map[string] Var);
}

// Convenience functions for creating new exported variables.

func NewInt(name string) *Int {
	v := new(Int);
	Publish(name, v);
	return v
}

func NewMap(name string) *Map {
	v := new(Map).Init();
	Publish(name, v);
	return v
}

func NewString(name string) *String {
	v := new(String);
	Publish(name, v);
	return v
}

// TODO(rsc): Make sure map access in separate thread is safe.
func iterate(c chan<- KeyValue) {
	for k, v := range vars {
		c <- KeyValue{ k, v };
	}
	close(c);
}

func Iter() <-chan KeyValue {
	c := make(chan KeyValue);
	go iterate(c);
	return c
}

func exvarHandler(c *http.Conn, req *http.Request) {
	c.SetHeader("content-type", "application/json; charset=utf-8");
	fmt.Fprintf(c, "{\n");
	first := true;
	for name, value := range vars {
		if !first {
			fmt.Fprintf(c, ",\n");
		}
		first = false;
		fmt.Fprintf(c, "  %q: %s", name, value);
	}
	fmt.Fprintf(c, "\n}\n");
}

func init() {
	http.Handle("/debug/vars", http.HandlerFunc(exvarHandler));
}
