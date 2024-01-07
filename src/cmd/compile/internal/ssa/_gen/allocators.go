// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// TODO: should we share backing storage for similarly-shaped types?
// e.g. []*Value and []*Block, or even []int32 and []bool.

import (
	"bytes"
	"fmt"
	"go/format"
	"io"
	"log"
	"os"
)

type allocator struct {
	name     string // name for alloc/free functions
	typ      string // the type they return/accept
	mak      string // code to make a new object (takes power-of-2 size as fmt arg)
	capacity string // code to calculate the capacity of an object. Should always report a power of 2.
	resize   string // code to shrink to sub-power-of-two size (takes size as fmt arg)
	clear    string // code for clearing object before putting it on the free list
	minLog   int    // log_2 of minimum allocation size
	maxLog   int    // log_2 of maximum allocation size
}

type derived struct {
	name string // name for alloc/free functions
	typ  string // the type they return/accept
	base string // underlying allocator
}

func genAllocators() {
	allocators := []allocator{
		{
			name:     "ValueSlice",
			typ:      "[]*Value",
			capacity: "cap(%s)",
			mak:      "make([]*Value, %s)",
			resize:   "%s[:%s]",
			clear:    "for i := range %[1]s {\n%[1]s[i] = nil\n}",
			minLog:   5,
			maxLog:   32,
		},
		{
			name:     "Int64Slice",
			typ:      "[]int64",
			capacity: "cap(%s)",
			mak:      "make([]int64, %s)",
			resize:   "%s[:%s]",
			clear:    "for i := range %[1]s {\n%[1]s[i] = 0\n}",
			minLog:   5,
			maxLog:   32,
		},
		{
			name:     "SparseSet",
			typ:      "*sparseSet",
			capacity: "%s.cap()",
			mak:      "newSparseSet(%s)",
			resize:   "", // larger-sized sparse sets are ok
			clear:    "%s.clear()",
			minLog:   5,
			maxLog:   32,
		},
		{
			name:     "SparseMap",
			typ:      "*sparseMap",
			capacity: "%s.cap()",
			mak:      "newSparseMap(%s)",
			resize:   "", // larger-sized sparse maps are ok
			clear:    "%s.clear()",
			minLog:   5,
			maxLog:   32,
		},
		{
			name:     "SparseMapPos",
			typ:      "*sparseMapPos",
			capacity: "%s.cap()",
			mak:      "newSparseMapPos(%s)",
			resize:   "", // larger-sized sparse maps are ok
			clear:    "%s.clear()",
			minLog:   5,
			maxLog:   32,
		},
	}
	deriveds := []derived{
		{
			name: "BlockSlice",
			typ:  "[]*Block",
			base: "ValueSlice",
		},
		{
			name: "IntSlice",
			typ:  "[]int",
			base: "Int64Slice",
		},
		{
			name: "Int32Slice",
			typ:  "[]int32",
			base: "Int64Slice",
		},
		{
			name: "Int8Slice",
			typ:  "[]int8",
			base: "Int64Slice",
		},
		{
			name: "BoolSlice",
			typ:  "[]bool",
			base: "Int64Slice",
		},
		{
			name: "IDSlice",
			typ:  "[]ID",
			base: "Int64Slice",
		},
	}

	w := new(bytes.Buffer)
	fmt.Fprintf(w, "// Code generated from _gen/allocators.go using 'go generate'; DO NOT EDIT.\n")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "package ssa")

	fmt.Fprintln(w, "import (")
	fmt.Fprintln(w, "\"internal/unsafeheader\"")
	fmt.Fprintln(w, "\"math/bits\"")
	fmt.Fprintln(w, "\"sync\"")
	fmt.Fprintln(w, "\"unsafe\"")
	fmt.Fprintln(w, ")")
	for _, a := range allocators {
		genAllocator(w, a)
	}
	for _, d := range deriveds {
		for _, base := range allocators {
			if base.name == d.base {
				genDerived(w, d, base)
				break
			}
		}
	}
	// gofmt result
	b := w.Bytes()
	var err error
	b, err = format.Source(b)
	if err != nil {
		fmt.Printf("%s\n", w.Bytes())
		panic(err)
	}

	if err := os.WriteFile("../allocators.go", b, 0666); err != nil {
		log.Fatalf("can't write output: %v\n", err)
	}
}
func genAllocator(w io.Writer, a allocator) {
	fmt.Fprintf(w, "var poolFree%s [%d]sync.Pool\n", a.name, a.maxLog-a.minLog)
	fmt.Fprintf(w, "func (c *Cache) alloc%s(n int) %s {\n", a.name, a.typ)
	fmt.Fprintf(w, "var s %s\n", a.typ)
	fmt.Fprintf(w, "n2 := n\n")
	fmt.Fprintf(w, "if n2 < %d { n2 = %d }\n", 1<<a.minLog, 1<<a.minLog)
	fmt.Fprintf(w, "b := bits.Len(uint(n2-1))\n")
	fmt.Fprintf(w, "v := poolFree%s[b-%d].Get()\n", a.name, a.minLog)
	fmt.Fprintf(w, "if v == nil {\n")
	fmt.Fprintf(w, "  s = %s\n", fmt.Sprintf(a.mak, "1<<b"))
	fmt.Fprintf(w, "} else {\n")
	if a.typ[0] == '*' {
		fmt.Fprintf(w, "s = v.(%s)\n", a.typ)
	} else {
		fmt.Fprintf(w, "sp := v.(*%s)\n", a.typ)
		fmt.Fprintf(w, "s = *sp\n")
		fmt.Fprintf(w, "*sp = nil\n")
		fmt.Fprintf(w, "c.hdr%s = append(c.hdr%s, sp)\n", a.name, a.name)
	}
	fmt.Fprintf(w, "}\n")
	if a.resize != "" {
		fmt.Fprintf(w, "s = %s\n", fmt.Sprintf(a.resize, "s", "n"))
	}
	fmt.Fprintf(w, "return s\n")
	fmt.Fprintf(w, "}\n")
	fmt.Fprintf(w, "func (c *Cache) free%s(s %s) {\n", a.name, a.typ)
	fmt.Fprintf(w, "%s\n", fmt.Sprintf(a.clear, "s"))
	fmt.Fprintf(w, "b := bits.Len(uint(%s) - 1)\n", fmt.Sprintf(a.capacity, "s"))
	if a.typ[0] == '*' {
		fmt.Fprintf(w, "poolFree%s[b-%d].Put(s)\n", a.name, a.minLog)
	} else {
		fmt.Fprintf(w, "var sp *%s\n", a.typ)
		fmt.Fprintf(w, "if len(c.hdr%s) == 0 {\n", a.name)
		fmt.Fprintf(w, "  sp = new(%s)\n", a.typ)
		fmt.Fprintf(w, "} else {\n")
		fmt.Fprintf(w, "  sp = c.hdr%s[len(c.hdr%s)-1]\n", a.name, a.name)
		fmt.Fprintf(w, "  c.hdr%s[len(c.hdr%s)-1] = nil\n", a.name, a.name)
		fmt.Fprintf(w, "  c.hdr%s = c.hdr%s[:len(c.hdr%s)-1]\n", a.name, a.name, a.name)
		fmt.Fprintf(w, "}\n")
		fmt.Fprintf(w, "*sp = s\n")
		fmt.Fprintf(w, "poolFree%s[b-%d].Put(sp)\n", a.name, a.minLog)
	}
	fmt.Fprintf(w, "}\n")
}
func genDerived(w io.Writer, d derived, base allocator) {
	fmt.Fprintf(w, "func (c *Cache) alloc%s(n int) %s {\n", d.name, d.typ)
	if d.typ[:2] != "[]" || base.typ[:2] != "[]" {
		panic(fmt.Sprintf("bad derived types: %s %s", d.typ, base.typ))
	}
	fmt.Fprintf(w, "var base %s\n", base.typ[2:])
	fmt.Fprintf(w, "var derived %s\n", d.typ[2:])
	fmt.Fprintf(w, "if unsafe.Sizeof(base)%%unsafe.Sizeof(derived) != 0 { panic(\"bad\") }\n")
	fmt.Fprintf(w, "scale := unsafe.Sizeof(base)/unsafe.Sizeof(derived)\n")
	fmt.Fprintf(w, "b := c.alloc%s(int((uintptr(n)+scale-1)/scale))\n", base.name)
	fmt.Fprintf(w, "s := unsafeheader.Slice {\n")
	fmt.Fprintf(w, "  Data: unsafe.Pointer(&b[0]),\n")
	fmt.Fprintf(w, "  Len: n,\n")
	fmt.Fprintf(w, "  Cap: cap(b)*int(scale),\n")
	fmt.Fprintf(w, "  }\n")
	fmt.Fprintf(w, "return *(*%s)(unsafe.Pointer(&s))\n", d.typ)
	fmt.Fprintf(w, "}\n")
	fmt.Fprintf(w, "func (c *Cache) free%s(s %s) {\n", d.name, d.typ)
	fmt.Fprintf(w, "var base %s\n", base.typ[2:])
	fmt.Fprintf(w, "var derived %s\n", d.typ[2:])
	fmt.Fprintf(w, "scale := unsafe.Sizeof(base)/unsafe.Sizeof(derived)\n")
	fmt.Fprintf(w, "b := unsafeheader.Slice {\n")
	fmt.Fprintf(w, "  Data: unsafe.Pointer(&s[0]),\n")
	fmt.Fprintf(w, "  Len: int((uintptr(len(s))+scale-1)/scale),\n")
	fmt.Fprintf(w, "  Cap: int((uintptr(cap(s))+scale-1)/scale),\n")
	fmt.Fprintf(w, "  }\n")
	fmt.Fprintf(w, "c.free%s(*(*%s)(unsafe.Pointer(&b)))\n", base.name, base.typ)
	fmt.Fprintf(w, "}\n")
}
