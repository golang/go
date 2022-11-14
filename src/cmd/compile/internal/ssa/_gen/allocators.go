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
			name:     "BlockSlice",
			typ:      "[]*Block",
			capacity: "cap(%s)",
			mak:      "make([]*Block, %s)",
			resize:   "%s[:%s]",
			clear:    "for i := range %[1]s {\n%[1]s[i] = nil\n}",
			minLog:   5,
			maxLog:   32,
		},
		{
			name:     "BoolSlice",
			typ:      "[]bool",
			capacity: "cap(%s)",
			mak:      "make([]bool, %s)",
			resize:   "%s[:%s]",
			clear:    "for i := range %[1]s {\n%[1]s[i] = false\n}",
			minLog:   8,
			maxLog:   32,
		},
		{
			name:     "IntSlice",
			typ:      "[]int",
			capacity: "cap(%s)",
			mak:      "make([]int, %s)",
			resize:   "%s[:%s]",
			clear:    "for i := range %[1]s {\n%[1]s[i] = 0\n}",
			minLog:   5,
			maxLog:   32,
		},
		{
			name:     "Int32Slice",
			typ:      "[]int32",
			capacity: "cap(%s)",
			mak:      "make([]int32, %s)",
			resize:   "%s[:%s]",
			clear:    "for i := range %[1]s {\n%[1]s[i] = 0\n}",
			minLog:   6,
			maxLog:   32,
		},
		{
			name:     "Int8Slice",
			typ:      "[]int8",
			capacity: "cap(%s)",
			mak:      "make([]int8, %s)",
			resize:   "%s[:%s]",
			clear:    "for i := range %[1]s {\n%[1]s[i] = 0\n}",
			minLog:   8,
			maxLog:   32,
		},
		{
			name:     "IDSlice",
			typ:      "[]ID",
			capacity: "cap(%s)",
			mak:      "make([]ID, %s)",
			resize:   "%s[:%s]",
			clear:    "for i := range %[1]s {\n%[1]s[i] = 0\n}",
			minLog:   6,
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

	w := new(bytes.Buffer)
	fmt.Fprintf(w, "// Code generated from _gen/allocators.go; DO NOT EDIT.\n")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "package ssa")

	fmt.Fprintln(w, "import (")
	fmt.Fprintln(w, "\"math/bits\"")
	fmt.Fprintln(w, "\"sync\"")
	fmt.Fprintln(w, ")")
	for _, a := range allocators {
		genAllocator(w, a)
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
