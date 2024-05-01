// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package expvar provides a standardized interface to public variables, such
// as operation counters in servers. It exposes these variables via HTTP at
// /debug/vars in JSON format. As of Go 1.22, the /debug/vars request must
// use GET.
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
// registering its HTTP handler and the above variables. To use it
// this way, link this package into your program:
//
//	import _ "expvar"
package expvar

import (
	"encoding/json"
	"internal/godebug"
	"log"
	"math"
	"net/http"
	"os"
	"runtime"
	"slices"
	"strconv"
	"sync"
	"sync/atomic"
	"unicode/utf8"
)

// Var is an abstract type for all exported variables.
type Var interface {
	// String returns a valid JSON value for the variable.
	// Types with String methods that do not return valid JSON
	// (such as time.Time) must not be used as a Var.
	String() string
}

type jsonVar interface {
	// appendJSON appends the JSON representation of the receiver to b.
	appendJSON(b []byte) []byte
}

// Int is a 64-bit integer variable that satisfies the [Var] interface.
type Int struct {
	i atomic.Int64
}

func (v *Int) Value() int64 {
	return v.i.Load()
}

func (v *Int) String() string {
	return string(v.appendJSON(nil))
}

func (v *Int) appendJSON(b []byte) []byte {
	return strconv.AppendInt(b, v.i.Load(), 10)
}

func (v *Int) Add(delta int64) {
	v.i.Add(delta)
}

func (v *Int) Set(value int64) {
	v.i.Store(value)
}

// Float is a 64-bit float variable that satisfies the [Var] interface.
type Float struct {
	f atomic.Uint64
}

func (v *Float) Value() float64 {
	return math.Float64frombits(v.f.Load())
}

func (v *Float) String() string {
	return string(v.appendJSON(nil))
}

func (v *Float) appendJSON(b []byte) []byte {
	return strconv.AppendFloat(b, math.Float64frombits(v.f.Load()), 'g', -1, 64)
}

// Add adds delta to v.
func (v *Float) Add(delta float64) {
	for {
		cur := v.f.Load()
		curVal := math.Float64frombits(cur)
		nxtVal := curVal + delta
		nxt := math.Float64bits(nxtVal)
		if v.f.CompareAndSwap(cur, nxt) {
			return
		}
	}
}

// Set sets v to value.
func (v *Float) Set(value float64) {
	v.f.Store(math.Float64bits(value))
}

// Map is a string-to-Var map variable that satisfies the [Var] interface.
type Map struct {
	m      sync.Map // map[string]Var
	keysMu sync.RWMutex
	keys   []string // sorted
}

// KeyValue represents a single entry in a [Map].
type KeyValue struct {
	Key   string
	Value Var
}

func (v *Map) String() string {
	return string(v.appendJSON(nil))
}

func (v *Map) appendJSON(b []byte) []byte {
	return v.appendJSONMayExpand(b, false)
}

func (v *Map) appendJSONMayExpand(b []byte, expand bool) []byte {
	afterCommaDelim := byte(' ')
	mayAppendNewline := func(b []byte) []byte { return b }
	if expand {
		afterCommaDelim = '\n'
		mayAppendNewline = func(b []byte) []byte { return append(b, '\n') }
	}

	b = append(b, '{')
	b = mayAppendNewline(b)
	first := true
	v.Do(func(kv KeyValue) {
		if !first {
			b = append(b, ',', afterCommaDelim)
		}
		first = false
		b = appendJSONQuote(b, kv.Key)
		b = append(b, ':', ' ')
		switch v := kv.Value.(type) {
		case nil:
			b = append(b, "null"...)
		case jsonVar:
			b = v.appendJSON(b)
		default:
			b = append(b, v.String()...)
		}
	})
	b = mayAppendNewline(b)
	b = append(b, '}')
	b = mayAppendNewline(b)
	return b
}

// Init removes all keys from the map.
func (v *Map) Init() *Map {
	v.keysMu.Lock()
	defer v.keysMu.Unlock()
	v.keys = v.keys[:0]
	v.m.Clear()
	return v
}

// addKey updates the sorted list of keys in v.keys.
func (v *Map) addKey(key string) {
	v.keysMu.Lock()
	defer v.keysMu.Unlock()
	// Using insertion sort to place key into the already-sorted v.keys.
	i, found := slices.BinarySearch(v.keys, key)
	if found {
		return
	}
	v.keys = slices.Insert(v.keys, i, key)
}

func (v *Map) Get(key string) Var {
	i, _ := v.m.Load(key)
	av, _ := i.(Var)
	return av
}

func (v *Map) Set(key string, av Var) {
	// Before we store the value, check to see whether the key is new. Try a Load
	// before LoadOrStore: LoadOrStore causes the key interface to escape even on
	// the Load path.
	if _, ok := v.m.Load(key); !ok {
		if _, dup := v.m.LoadOrStore(key, av); !dup {
			v.addKey(key)
			return
		}
	}

	v.m.Store(key, av)
}

// Add adds delta to the *[Int] value stored under the given map key.
func (v *Map) Add(key string, delta int64) {
	i, ok := v.m.Load(key)
	if !ok {
		var dup bool
		i, dup = v.m.LoadOrStore(key, new(Int))
		if !dup {
			v.addKey(key)
		}
	}

	// Add to Int; ignore otherwise.
	if iv, ok := i.(*Int); ok {
		iv.Add(delta)
	}
}

// AddFloat adds delta to the *[Float] value stored under the given map key.
func (v *Map) AddFloat(key string, delta float64) {
	i, ok := v.m.Load(key)
	if !ok {
		var dup bool
		i, dup = v.m.LoadOrStore(key, new(Float))
		if !dup {
			v.addKey(key)
		}
	}

	// Add to Float; ignore otherwise.
	if iv, ok := i.(*Float); ok {
		iv.Add(delta)
	}
}

// Delete deletes the given key from the map.
func (v *Map) Delete(key string) {
	v.keysMu.Lock()
	defer v.keysMu.Unlock()
	i, found := slices.BinarySearch(v.keys, key)
	if found {
		v.keys = slices.Delete(v.keys, i, i+1)
		v.m.Delete(key)
	}
}

// Do calls f for each entry in the map.
// The map is locked during the iteration,
// but existing entries may be concurrently updated.
func (v *Map) Do(f func(KeyValue)) {
	v.keysMu.RLock()
	defer v.keysMu.RUnlock()
	for _, k := range v.keys {
		i, _ := v.m.Load(k)
		val, _ := i.(Var)
		f(KeyValue{k, val})
	}
}

// String is a string variable, and satisfies the [Var] interface.
type String struct {
	s atomic.Value // string
}

func (v *String) Value() string {
	p, _ := v.s.Load().(string)
	return p
}

// String implements the [Var] interface. To get the unquoted string
// use [String.Value].
func (v *String) String() string {
	return string(v.appendJSON(nil))
}

func (v *String) appendJSON(b []byte) []byte {
	return appendJSONQuote(b, v.Value())
}

func (v *String) Set(value string) {
	v.s.Store(value)
}

// Func implements [Var] by calling the function
// and formatting the returned value using JSON.
type Func func() any

func (f Func) Value() any {
	return f()
}

func (f Func) String() string {
	v, _ := json.Marshal(f())
	return string(v)
}

// All published variables.
var vars Map

// Publish declares a named exported variable. This should be called from a
// package's init function when it creates its Vars. If the name is already
// registered then this will log.Panic.
func Publish(name string, v Var) {
	if _, dup := vars.m.LoadOrStore(name, v); dup {
		log.Panicln("Reuse of exported var name:", name)
	}
	vars.keysMu.Lock()
	defer vars.keysMu.Unlock()
	vars.keys = append(vars.keys, name)
	slices.Sort(vars.keys)
}

// Get retrieves a named exported variable. It returns nil if the name has
// not been registered.
func Get(name string) Var {
	return vars.Get(name)
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
	vars.Do(f)
}

func expvarHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.Write(vars.appendJSONMayExpand(nil, true))
}

// Handler returns the expvar HTTP Handler.
//
// This is only needed to install the handler in a non-standard location.
func Handler() http.Handler {
	return http.HandlerFunc(expvarHandler)
}

func cmdline() any {
	return os.Args
}

func memstats() any {
	stats := new(runtime.MemStats)
	runtime.ReadMemStats(stats)
	return *stats
}

func init() {
	if godebug.New("httpmuxgo121").Value() == "1" {
		http.HandleFunc("/debug/vars", expvarHandler)
	} else {
		http.HandleFunc("GET /debug/vars", expvarHandler)
	}
	Publish("cmdline", Func(cmdline))
	Publish("memstats", Func(memstats))
}

// TODO: Use json.appendString instead.
func appendJSONQuote(b []byte, s string) []byte {
	const hex = "0123456789abcdef"
	b = append(b, '"')
	for _, r := range s {
		switch {
		case r < ' ' || r == '\\' || r == '"' || r == '<' || r == '>' || r == '&' || r == '\u2028' || r == '\u2029':
			switch r {
			case '\\', '"':
				b = append(b, '\\', byte(r))
			case '\n':
				b = append(b, '\\', 'n')
			case '\r':
				b = append(b, '\\', 'r')
			case '\t':
				b = append(b, '\\', 't')
			default:
				b = append(b, '\\', 'u', hex[(r>>12)&0xf], hex[(r>>8)&0xf], hex[(r>>4)&0xf], hex[(r>>0)&0xf])
			}
		case r < utf8.RuneSelf:
			b = append(b, byte(r))
		default:
			b = utf8.AppendRune(b, r)
		}
	}
	b = append(b, '"')
	return b
}
