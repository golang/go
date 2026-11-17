// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"
	"reflect"
	"runtime"
	"runtime/debug"
	"sync/atomic"
	"unsafe"
	"weak"
)

const (
	rounds     = 50
	listLen    = 3000
	mapLen     = 1500
	ifaceLen   = 1500
	chainLen   = 12
	weakCount  = 64
	finalCount = 64
)

var (
	failed    bool
	finalized atomic.Uint64
)

func emit(name string, ok bool, detail string) {
	if ok {
		fmt.Printf("%s ok %s\n", name, detail)
		return
	}
	fmt.Printf("LAW %s %s\n", name, detail)
	failed = true
}

func mix(a, b uint64) uint64 {
	return a*6364136223846793005 + b + 1442695040888963407
}

type node struct {
	next *node
	id   uint64
	sum  uint64
}

func buildList(n int, seed uint64) *node {
	var head *node
	for i := range n {
		var below uint64
		if head != nil {
			below = head.sum
		}
		id := seed + uint64(i)
		head = &node{next: head, id: id, sum: mix(below, id)}
	}
	return head
}

func listIntact(head *node, n int) bool {
	count := 0
	for p := head; p != nil; p = p.next {
		var below uint64
		if p.next != nil {
			below = p.next.sum
		}
		if p.sum != mix(below, p.id) {
			return false
		}
		count++
	}
	return count == n
}

type item struct {
	self *item
	key  int
	data [2]uint64
	sum  uint64
}

func newItem(key int) *item {
	it := &item{key: key, data: [2]uint64{mix(uint64(key), 1), mix(uint64(key), 2)}}
	it.self = it
	it.sum = mix(it.data[0], it.data[1])
	return it
}

func itemIntact(it *item) bool {
	return it != nil && it.self == it && it.sum == mix(it.data[0], it.data[1]) && it.data[0] == mix(uint64(it.key), 1)
}

func buildMap(n int) map[int]*item {
	m := make(map[int]*item, n)
	for i := range n {
		m[i] = newItem(i)
	}
	return m
}

func mapIntact(m map[int]*item, n int) bool {
	if len(m) != n {
		return false
	}
	for k, it := range m {
		if !itemIntact(it) || it.key != k {
			return false
		}
	}
	return true
}

type pair struct {
	left, right *item
	tag         uint64
}

func buildIfaces(n int) []any {
	out := make([]any, 0, n)
	for i := range n {
		switch i % 5 {
		case 0:
			out = append(out, newItem(i))
		case 1:
			out = append(out, &node{id: uint64(i), sum: mix(0, uint64(i))})
		case 2:
			out = append(out, fmt.Sprint("s", i))
		case 3:
			out = append(out, pair{left: newItem(i), right: newItem(i + 1), tag: uint64(i)})
		default:
			x := newItem(i)
			out = append(out, func() *item { return x })
		}
	}
	return out
}

func ifacesIntact(vs []any, n int) bool {
	if len(vs) != n {
		return false
	}
	for i, v := range vs {
		switch v := v.(type) {
		case *item:
			if i%5 != 0 || !itemIntact(v) || v.key != i {
				return false
			}
		case *node:
			if i%5 != 1 || v.sum != mix(0, uint64(i)) {
				return false
			}
		case string:
			if i%5 != 2 || v != fmt.Sprint("s", i) {
				return false
			}
		case pair:
			if i%5 != 3 || !itemIntact(v.left) || !itemIntact(v.right) || v.left.key != i || v.right.key != i+1 || v.tag != uint64(i) {
				return false
			}
		case func() *item:
			if i%5 != 4 || !itemIntact(v()) || v().key != i {
				return false
			}
		default:
			return false
		}
	}
	return true
}

var bigPads = []int{500, 1100, 2600, 5200, 9000, 16400, 24600, 32000}

type bigShape struct {
	typ  reflect.Type
	head unsafe.Pointer
}

func bigType(pad int) reflect.Type {
	return reflect.StructOf([]reflect.StructField{
		{Name: "Next", Type: reflect.TypeFor[unsafe.Pointer]()},
		{Name: "Prev", Type: reflect.TypeFor[unsafe.Pointer]()},
		{Name: "Sum", Type: reflect.TypeFor[uint64]()},
		{Name: "Pad", Type: reflect.ArrayOf(pad, reflect.TypeFor[byte]())},
	})
}

func bigSum(pad int, i int, first, last byte) uint64 {
	return mix(mix(uint64(pad), uint64(i)), uint64(first)<<8|uint64(last))
}

func buildBig(pad int) bigShape {
	typ := bigType(pad)
	var head unsafe.Pointer
	for i := range chainLen {
		p := reflect.New(typ)
		v := p.Elem()
		first, last := byte(i+1), byte(255-i)
		v.Field(3).Index(0).SetUint(uint64(first))
		v.Field(3).Index(pad - 1).SetUint(uint64(last))
		v.Field(2).SetUint(bigSum(pad, i, first, last))
		v.Field(0).SetPointer(head)
		if head != nil {
			reflect.NewAt(typ, head).Elem().Field(1).SetPointer(p.UnsafePointer())
		}
		head = p.UnsafePointer()
	}
	return bigShape{typ: typ, head: head}
}

func bigIntact(b bigShape, pad int) bool {
	count := 0
	var prev unsafe.Pointer
	for p := b.head; p != nil; {
		v := reflect.NewAt(b.typ, p).Elem()
		i := chainLen - 1 - count
		first, last := byte(i+1), byte(255-i)
		pads := v.Field(3)
		if byte(pads.Index(0).Uint()) != first || byte(pads.Index(pad-1).Uint()) != last {
			return false
		}
		if v.Field(2).Uint() != bigSum(pad, i, first, last) {
			return false
		}
		if v.Field(1).UnsafePointer() != prev && count > 0 {
			return false
		}
		prev = p
		p = v.Field(0).UnsafePointer()
		count++
	}
	return count == chainLen
}

type world struct {
	list    *node
	items   map[int]*item
	ifaces  []any
	bigs    []bigShape
	weaks   []weak.Pointer[node]
	pinned  []*node
	garbage any
}

func newWorld() *world {
	w := &world{
		list:   buildList(listLen, 7),
		items:  buildMap(mapLen),
		ifaces: buildIfaces(ifaceLen),
	}
	for _, pad := range bigPads {
		w.bigs = append(w.bigs, buildBig(pad))
	}
	p := w.list
	for i := 0; i < weakCount && p != nil; i++ {
		w.weaks = append(w.weaks, weak.Make(p))
		w.pinned = append(w.pinned, p)
		p = p.next
	}
	p = w.list
	for i := 0; i < finalCount && p != nil; i++ {
		runtime.AddCleanup(p, func(uint64) { finalized.Add(1) }, p.id)
		p = p.next
	}
	return w
}

func (w *world) churn(round int) {
	g := buildList(listLen/2, uint64(round))
	m := buildMap(mapLen / 2)
	v := buildIfaces(ifaceLen / 2)
	var bigs []bigShape
	for _, pad := range bigPads {
		bigs = append(bigs, buildBig(pad))
	}
	w.garbage = []any{g, m, v, bigs}
	w.garbage = nil
}

func (w *world) intact() (shape string, ok bool) {
	switch {
	case !listIntact(w.list, listLen):
		return "list", false
	case !mapIntact(w.items, mapLen):
		return "map", false
	case !ifacesIntact(w.ifaces, ifaceLen):
		return "ifaces", false
	}
	for i, b := range w.bigs {
		if !bigIntact(b, bigPads[i]) {
			return fmt.Sprint("big", bigPads[i]), false
		}
	}
	for i, wp := range w.weaks {
		if wp.Value() != w.pinned[i] {
			return "weak", false
		}
	}
	return "", true
}

func main() {
	debug.SetGCPercent(1)
	w := newWorld()
	for round := range rounds {
		w.churn(round)
		runtime.GC()
		if shape, ok := w.intact(); !ok {
			emit("round", false, fmt.Sprint("round=", round, " shape=", shape))
			os.Exit(1)
		}
	}
	emit("list", listIntact(w.list, listLen), fmt.Sprint("len=", listLen, " head=", w.list.sum))
	emit("map", mapIntact(w.items, mapLen), fmt.Sprint("len=", mapLen))
	emit("ifaces", ifacesIntact(w.ifaces, ifaceLen), fmt.Sprint("len=", ifaceLen))
	for i, b := range w.bigs {
		emit(fmt.Sprint("big", bigPads[i]), bigIntact(b, bigPads[i]), fmt.Sprint("chain=", chainLen))
	}
	live := 0
	for i, wp := range w.weaks {
		if wp.Value() == w.pinned[i] {
			live++
		}
	}
	emit("weak", live == weakCount, fmt.Sprint("live=", live))
	runtime.GC()
	runtime.Gosched()
	emit("finalizers", finalized.Load() == 0, fmt.Sprint("ran=", finalized.Load()))
	runtime.KeepAlive(w)
	if failed {
		os.Exit(1)
	}
	fmt.Println("ok")
}
