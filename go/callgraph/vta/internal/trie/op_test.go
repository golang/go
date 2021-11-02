// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trie_test

import (
	"fmt"
	"math/rand"
	"reflect"
	"testing"
	"time"

	"golang.org/x/tools/go/callgraph/vta/internal/trie"
)

// This file tests trie.Map by cross checking operations on a collection of
// trie.Map's against a collection of map[uint64]interface{}. This includes
// both limited fuzz testing for correctness and benchmarking.

// mapCollection is effectively a []map[uint64]interface{}.
// These support operations being applied to the i'th maps.
type mapCollection interface {
	Elements() []map[uint64]interface{}

	DeepEqual(l, r int) bool
	Lookup(id int, k uint64) (interface{}, bool)

	Insert(id int, k uint64, v interface{})
	Update(id int, k uint64, v interface{})
	Remove(id int, k uint64)
	Intersect(l int, r int)
	Merge(l int, r int)
	Clear(id int)

	Average(l int, r int)
	Assign(l int, r int)
}

// opCode of an operation.
type opCode int

const (
	deepEqualsOp opCode = iota
	lookupOp
	insert
	update
	remove
	merge
	intersect
	clear
	takeAverage
	assign
)

func (op opCode) String() string {
	switch op {
	case deepEqualsOp:
		return "DE"
	case lookupOp:
		return "LO"
	case insert:
		return "IN"
	case update:
		return "UP"
	case remove:
		return "RE"
	case merge:
		return "ME"
	case intersect:
		return "IT"
	case clear:
		return "CL"
	case takeAverage:
		return "AV"
	case assign:
		return "AS"
	default:
		return "??"
	}
}

// A mapCollection backed by MutMaps.
type trieCollection struct {
	b     *trie.Builder
	tries []trie.MutMap
}

func (c *trieCollection) Elements() []map[uint64]interface{} {
	var maps []map[uint64]interface{}
	for _, m := range c.tries {
		maps = append(maps, trie.Elems(m.M))
	}
	return maps
}
func (c *trieCollection) Eq(id int, m map[uint64]interface{}) bool {
	elems := trie.Elems(c.tries[id].M)
	return !reflect.DeepEqual(elems, m)
}

func (c *trieCollection) Lookup(id int, k uint64) (interface{}, bool) {
	return c.tries[id].M.Lookup(k)
}
func (c *trieCollection) DeepEqual(l, r int) bool {
	return c.tries[l].M.DeepEqual(c.tries[r].M)
}

func (c *trieCollection) Add() {
	c.tries = append(c.tries, c.b.MutEmpty())
}

func (c *trieCollection) Insert(id int, k uint64, v interface{}) {
	c.tries[id].Insert(k, v)
}

func (c *trieCollection) Update(id int, k uint64, v interface{}) {
	c.tries[id].Update(k, v)
}

func (c *trieCollection) Remove(id int, k uint64) {
	c.tries[id].Remove(k)
}

func (c *trieCollection) Intersect(l int, r int) {
	c.tries[l].Intersect(c.tries[r].M)
}

func (c *trieCollection) Merge(l int, r int) {
	c.tries[l].Merge(c.tries[r].M)
}

func (c *trieCollection) Average(l int, r int) {
	c.tries[l].MergeWith(average, c.tries[r].M)
}

func (c *trieCollection) Clear(id int) {
	c.tries[id] = c.b.MutEmpty()
}
func (c *trieCollection) Assign(l, r int) {
	c.tries[l] = c.tries[r]
}

func average(x interface{}, y interface{}) interface{} {
	if x, ok := x.(float32); ok {
		if y, ok := y.(float32); ok {
			return (x + y) / 2.0
		}
	}
	return x
}

type builtinCollection []map[uint64]interface{}

func (c builtinCollection) Elements() []map[uint64]interface{} {
	return c
}

func (c builtinCollection) Lookup(id int, k uint64) (interface{}, bool) {
	v, ok := c[id][k]
	return v, ok
}
func (c builtinCollection) DeepEqual(l, r int) bool {
	return reflect.DeepEqual(c[l], c[r])
}

func (c builtinCollection) Insert(id int, k uint64, v interface{}) {
	if _, ok := c[id][k]; !ok {
		c[id][k] = v
	}
}

func (c builtinCollection) Update(id int, k uint64, v interface{}) {
	c[id][k] = v
}

func (c builtinCollection) Remove(id int, k uint64) {
	delete(c[id], k)
}

func (c builtinCollection) Intersect(l int, r int) {
	result := map[uint64]interface{}{}
	for k, v := range c[l] {
		if _, ok := c[r][k]; ok {
			result[k] = v
		}
	}
	c[l] = result
}

func (c builtinCollection) Merge(l int, r int) {
	result := map[uint64]interface{}{}
	for k, v := range c[r] {
		result[k] = v
	}
	for k, v := range c[l] {
		result[k] = v
	}
	c[l] = result
}

func (c builtinCollection) Average(l int, r int) {
	avg := map[uint64]interface{}{}
	for k, lv := range c[l] {
		if rv, ok := c[r][k]; ok {
			avg[k] = average(lv, rv)
		} else {
			avg[k] = lv // add elements just in l
		}
	}
	for k, rv := range c[r] {
		if _, ok := c[l][k]; !ok {
			avg[k] = rv // add elements just in r
		}
	}
	c[l] = avg
}

func (c builtinCollection) Assign(l, r int) {
	m := map[uint64]interface{}{}
	for k, v := range c[r] {
		m[k] = v
	}
	c[l] = m
}

func (c builtinCollection) Clear(id int) {
	c[id] = map[uint64]interface{}{}
}

func newTriesCollection(size int) *trieCollection {
	tc := &trieCollection{
		b:     trie.NewBuilder(),
		tries: make([]trie.MutMap, size),
	}
	for i := 0; i < size; i++ {
		tc.tries[i] = tc.b.MutEmpty()
	}
	return tc
}

func newMapsCollection(size int) *builtinCollection {
	maps := make(builtinCollection, size)
	for i := 0; i < size; i++ {
		maps[i] = map[uint64]interface{}{}
	}
	return &maps
}

// operation on a map collection.
type operation struct {
	code opCode
	l, r int
	k    uint64
	v    float32
}

// Apply the operation to maps.
func (op operation) Apply(maps mapCollection) interface{} {
	type lookupresult struct {
		v  interface{}
		ok bool
	}
	switch op.code {
	case deepEqualsOp:
		return maps.DeepEqual(op.l, op.r)
	case lookupOp:
		v, ok := maps.Lookup(op.l, op.k)
		return lookupresult{v, ok}
	case insert:
		maps.Insert(op.l, op.k, op.v)
	case update:
		maps.Update(op.l, op.k, op.v)
	case remove:
		maps.Remove(op.l, op.k)
	case merge:
		maps.Merge(op.l, op.r)
	case intersect:
		maps.Intersect(op.l, op.r)
	case clear:
		maps.Clear(op.l)
	case takeAverage:
		maps.Average(op.l, op.r)
	case assign:
		maps.Assign(op.l, op.r)
	}
	return nil
}

// Returns a collection of op codes with dist[op] copies of op.
func distribution(dist map[opCode]int) []opCode {
	var codes []opCode
	for op, n := range dist {
		for i := 0; i < n; i++ {
			codes = append(codes, op)
		}
	}
	return codes
}

// options for generating a random operation.
type options struct {
	maps   int
	maxKey uint64
	maxVal int
	codes  []opCode
}

// returns a random operation using r as a source of randomness.
func randOperator(r *rand.Rand, opts options) operation {
	id := func() int { return r.Intn(opts.maps) }
	key := func() uint64 { return r.Uint64() % opts.maxKey }
	val := func() float32 { return float32(r.Intn(opts.maxVal)) }
	switch code := opts.codes[r.Intn(len(opts.codes))]; code {
	case lookupOp, remove:
		return operation{code: code, l: id(), k: key()}
	case insert, update:
		return operation{code: code, l: id(), k: key(), v: val()}
	case deepEqualsOp, merge, intersect, takeAverage, assign:
		return operation{code: code, l: id(), r: id()}
	case clear:
		return operation{code: code, l: id()}
	default:
		panic("Invalid op code")
	}
}

func randOperators(r *rand.Rand, numops int, opts options) []operation {
	ops := make([]operation, numops)
	for i := 0; i < numops; i++ {
		ops[i] = randOperator(r, opts)
	}
	return ops
}

// TestOperations applies a series of random operations to collection of
// trie.MutMaps and map[uint64]interface{}. It tests for the maps being equal.
func TestOperations(t *testing.T) {
	seed := time.Now().UnixNano()
	s := rand.NewSource(seed)
	r := rand.New(s)
	t.Log("seed: ", seed)

	size := 10
	N := 100000
	ops := randOperators(r, N, options{
		maps:   size,
		maxKey: 128,
		maxVal: 100,
		codes: distribution(map[opCode]int{
			deepEqualsOp: 1,
			lookupOp:     10,
			insert:       10,
			update:       10,
			remove:       10,
			merge:        10,
			intersect:    10,
			clear:        2,
			takeAverage:  5,
			assign:       5,
		}),
	})

	var tries mapCollection = newTriesCollection(size)
	var maps mapCollection = newMapsCollection(size)
	check := func() error {
		if got, want := tries.Elements(), maps.Elements(); !reflect.DeepEqual(got, want) {
			return fmt.Errorf("elements of tries and maps and tries differed. got %v want %v", got, want)
		}
		return nil
	}

	for i, op := range ops {
		got, want := op.Apply(tries), op.Apply(maps)
		if got != want {
			t.Errorf("op[%d]: (%v).Apply(%v) != (%v).Apply(%v). got %v want %v",
				i, op, tries, op, maps, got, want)
		}
	}
	if err := check(); err != nil {
		t.Errorf("%d operators failed with %s", size, err)
		t.Log("Rerunning with more checking")
		tries, maps = newTriesCollection(size), newMapsCollection(size)
		for i, op := range ops {
			op.Apply(tries)
			op.Apply(maps)
			if err := check(); err != nil {
				t.Fatalf("Failed first on op[%d]=%v: %v", i, op, err)
			}
		}
	}
}

func run(b *testing.B, opts options, seed int64, mk func(int) mapCollection) {
	r := rand.New(rand.NewSource(seed))
	ops := randOperators(r, b.N, opts)
	maps := mk(opts.maps)
	for _, op := range ops {
		op.Apply(maps)
	}
}

var standard options = options{
	maps:   10,
	maxKey: 128,
	maxVal: 100,
	codes: distribution(map[opCode]int{
		deepEqualsOp: 1,
		lookupOp:     20,
		insert:       20,
		update:       20,
		remove:       20,
		merge:        10,
		intersect:    10,
		clear:        1,
		takeAverage:  5,
		assign:       20,
	}),
}

func BenchmarkTrieStandard(b *testing.B) {
	run(b, standard, 123, func(size int) mapCollection {
		return newTriesCollection(size)
	})
}

func BenchmarkMapsStandard(b *testing.B) {
	run(b, standard, 123, func(size int) mapCollection {
		return newMapsCollection(size)
	})
}

var smallWide options = options{
	maps:   100,
	maxKey: 100,
	maxVal: 8,
	codes: distribution(map[opCode]int{
		deepEqualsOp: 0,
		lookupOp:     0,
		insert:       30,
		update:       20,
		remove:       0,
		merge:        10,
		intersect:    0,
		clear:        1,
		takeAverage:  0,
		assign:       30,
	}),
}

func BenchmarkTrieSmallWide(b *testing.B) {
	run(b, smallWide, 456, func(size int) mapCollection {
		return newTriesCollection(size)
	})
}

func BenchmarkMapsSmallWide(b *testing.B) {
	run(b, smallWide, 456, func(size int) mapCollection {
		return newMapsCollection(size)
	})
}
