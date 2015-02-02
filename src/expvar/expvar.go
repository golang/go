// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package expvar provides a standardized interface to public variables, such
// as operation counters in servers. It exposes these variables via HTTP at
// /debug/vars in JSON format.
//
// Operations to set or modify these public variables are atomic.
//
// In addition to adding the HTTP handler, this package registers the
// following variables:
//
//	cmdline   os.Args
//	memstats  runtime.Memstats
//
// The package is sometimes only imported for the side effect of
// registering its HTTP handler and the above variables.  To use it
// this way, link this package into your program:
//	import _ "expvar"
//
package expvar

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"runtime"
	"sort"
	"strconv"
	"sync"
	"sync/atomic"
)

// Var is an abstract type for all exported variables.
type Var interface {
	String() string
}

// Int is a 64-bit integer variable that satisfies the Var interface.
type Int struct {
	i int64
}

func (v *Int) String() string {
	return strconv.FormatInt(atomic.LoadInt64(&v.i), 10)
}

func (v *Int) Add(delta int64) {
	atomic.AddInt64(&v.i, delta)
}

func (v *Int) Set(value int64) {
	atomic.StoreInt64(&v.i, value)
}

// Float is a 64-bit float variable that satisfies the Var interface.
type Float struct {
	f uint64
}

func (v *Float) String() string {
	return strconv.FormatFloat(
		math.Float64frombits(atomic.LoadUint64(&v.f)), 'g', -1, 64)
}

// Add adds delta to v.
func (v *Float) Add(delta float64) {
	for {
		cur := atomic.LoadUint64(&v.f)
		curVal := math.Float64frombits(cur)
		nxtVal := curVal + delta
		nxt := math.Float64bits(nxtVal)
		if atomic.CompareAndSwapUint64(&v.f, cur, nxt) {
			return
		}
	}
}

// Set sets v to value.
func (v *Float) Set(value float64) {
	atomic.StoreUint64(&v.f, math.Float64bits(value))
}

// Map is a string-to-Var map variable that satisfies the Var interface.
type Map struct {
	mu   sync.RWMutex
	m    map[string]Var
	keys []string // sorted
}

// KeyValue represents a single entry in a Map.
type KeyValue struct {
	Key   string
	Value Var
}

func (v *Map) String() string {
	v.mu.RLock()
	defer v.mu.RUnlock()
	var b bytes.Buffer
	fmt.Fprintf(&b, "{")
	first := true
	v.doLocked(func(kv KeyValue) {
		if !first {
			fmt.Fprintf(&b, ", ")
		}
		fmt.Fprintf(&b, "%q: %v", kv.Key, kv.Value)
		first = false
	})
	fmt.Fprintf(&b, "}")
	return b.String()
}

func (v *Map) Init() *Map {
	v.m = make(map[string]Var)
	return v
}

// updateKeys updates the sorted list of keys in v.keys.
// must be called with v.mu held.
func (v *Map) updateKeys() {
	if len(v.m) == len(v.keys) {
		// No new key.
		return
	}
	v.keys = v.keys[:0]
	for k := range v.m {
		v.keys = append(v.keys, k)
	}
	sort.Strings(v.keys)
}

func (v *Map) Get(key string) Var {
	v.mu.RLock()
	defer v.mu.RUnlock()
	return v.m[key]
}

func (v *Map) Set(key string, av Var) {
	v.mu.Lock()
	defer v.mu.Unlock()
	v.m[key] = av
	v.updateKeys()
}

func (v *Map) Add(key string, delta int64) {
	v.mu.RLock()
	av, ok := v.m[key]
	v.mu.RUnlock()
	if !ok {
		// check again under the write lock
		v.mu.Lock()
		av, ok = v.m[key]
		if !ok {
			av = new(Int)
			v.m[key] = av
			v.updateKeys()
		}
		v.mu.Unlock()
	}

	// Add to Int; ignore otherwise.
	if iv, ok := av.(*Int); ok {
		iv.Add(delta)
	}
}

// AddFloat adds delta to the *Float value stored under the given map key.
func (v *Map) AddFloat(key string, delta float64) {
	v.mu.RLock()
	av, ok := v.m[key]
	v.mu.RUnlock()
	if !ok {
		// check again under the write lock
		v.mu.Lock()
		av, ok = v.m[key]
		if !ok {
			av = new(Float)
			v.m[key] = av
			v.updateKeys()
		}
		v.mu.Unlock()
	}

	// Add to Float; ignore otherwise.
	if iv, ok := av.(*Float); ok {
		iv.Add(delta)
	}
}

// Do calls f for each entry in the map.
// The map is locked during the iteration,
// but existing entries may be concurrently updated.
func (v *Map) Do(f func(KeyValue)) {
	v.mu.RLock()
	defer v.mu.RUnlock()
	v.doLocked(f)
}

// doLocked calls f for each entry in the map.
// v.mu must be held for reads.
func (v *Map) doLocked(f func(KeyValue)) {
	for _, k := range v.keys {
		f(KeyValue{k, v.m[k]})
	}
}

// String is a string variable, and satisfies the Var interface.
type String struct {
	mu sync.RWMutex
	s  string
}

func (v *String) String() string {
	v.mu.RLock()
	defer v.mu.RUnlock()
	return strconv.Quote(v.s)
}

func (v *String) Set(value string) {
	v.mu.Lock()
	defer v.mu.Unlock()
	v.s = value
}

// Func implements Var by calling the function
// and formatting the returned value using JSON.
type Func func() interface{}

func (f Func) String() string {
	v, _ := json.Marshal(f())
	return string(v)
}

// All published variables.
var (
	mutex   sync.RWMutex
	vars    = make(map[string]Var)
	varKeys []string // sorted
)

// Publish declares a named exported variable. This should be called from a
// package's init function when it creates its Vars. If the name is already
// registered then this will log.Panic.
func Publish(name string, v Var) {
	mutex.Lock()
	defer mutex.Unlock()
	if _, existing := vars[name]; existing {
		log.Panicln("Reuse of exported var name:", name)
	}
	vars[name] = v
	varKeys = append(varKeys, name)
	sort.Strings(varKeys)
}

// Get retrieves a named exported variable.
func Get(name string) Var {
	mutex.RLock()
	defer mutex.RUnlock()
	return vars[name]
}

// Convenience functions for creating new exported variables.

func NewInt(name string) *Int {
	v := new(Int)
	Publish(name, v)
	return v
}

func NewFloat(name string) *Float {
	v := new(Float)
	Publish(name, v)
	return v
}

func NewMap(name string) *Map {
	v := new(Map).Init()
	Publish(name, v)
	return v
}

func NewString(name string) *String {
	v := new(String)
	Publish(name, v)
	return v
}

// Do calls f for each exported variable.
// The global variable map is locked during the iteration,
// but existing entries may be concurrently updated.
func Do(f func(KeyValue)) {
	mutex.RLock()
	defer mutex.RUnlock()
	for _, k := range varKeys {
		f(KeyValue{k, vars[k]})
	}
}

func expvarHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	fmt.Fprintf(w, "{\n")
	first := true
	Do(func(kv KeyValue) {
		if !first {
			fmt.Fprintf(w, ",\n")
		}
		first = false
		fmt.Fprintf(w, "%q: %s", kv.Key, kv.Value)
	})
	fmt.Fprintf(w, "\n}\n")
}

func cmdline() interface{} {
	return os.Args
}

func memstats() interface{} {
	stats := new(runtime.MemStats)
	runtime.ReadMemStats(stats)
	return *stats
}

func init() {
	http.HandleFunc("/debug/vars", expvarHandler)
	Publish("cmdline", Func(cmdline))
	Publish("memstats", Func(memstats))
}
