// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gen

import (
	"container/heap"
	"encoding/binary"
	"fmt"
	"hash/maphash"
	"io"
	"log"
	"os"
	"reflect"
	"strings"
)

const logCompile = true

func fatalf(f string, args ...any) {
	panic(fmt.Sprintf(f, args...))
}

type File struct {
	w      io.Writer
	funcs  []*Func
	consts []fileConst
}

func NewFile(w io.Writer) *File {
	return &File{w: w}
}

func (f *File) AddFunc(fn *Func) {
	f.funcs = append(f.funcs, fn)
}

type fileConst struct {
	name string
	data any
}

func (f *File) AddConst(name string, data any) {
	// TODO: It would be nice if this were unified with "const" ops, but the
	// reason I added this was for []*Func consts, which would take an overhaul
	// to represent in "const" ops.
	f.consts = append(f.consts, fileConst{name, data})
}

type Func struct {
	name  string
	nArgs int
	idGen int
	ops   []*op
}

func NewFunc(name string) *Func {
	fn := &Func{name: name}
	return fn
}

// attach adds x to fn's op list. If x has any unattached arguments, this adds
// those first (recursively).
func (fn *Func) attach(x *op) {
	// Make sure the arguments are attached to the function.
	for _, arg := range x.args {
		argFn := arg.fn
		if argFn == nil {
			fn.attach(arg)
		} else if argFn != fn {
			panic("ops from different functions")
		}
	}

	x.fn = fn
	x.id = fn.idGen
	fn.idGen++
	fn.ops = append(fn.ops, x)
}

func Arg[W wrap[T], T Word](fn *Func) T {
	loc := locReg{cls: regClassGP, reg: fn.nArgs}
	fn.nArgs++
	var x W
	o := &op{op: "arg", kind: x.kind(), c: loc}
	fn.attach(o)
	return x.wrap(o)
}

func Return(results ...Value) {
	args := make([]*op, len(results))
	for i, res := range results {
		args[i] = res.getOp()
	}
	var x void
	x.initOp(&op{op: "return", kind: voidKind, args: args})
}

type op struct {
	op   string
	kind *kind
	args []*op

	id int
	fn *Func

	// c depends on "op".
	//
	// arg locReg - The register containing the argument value
	// const any  - The constant value
	// deref int  - Byte offset from args[0]
	c    any
	name string
}

func (o *op) String() string {
	return fmt.Sprintf("v%02d", o.id)
}

func imm(val any) *op {
	return &op{op: "imm", c: val}
}

func (o *op) equalNoName(o2 *op) bool {
	if o.op != o2.op || o.c != o2.c || len(o.args) != len(o2.args) {
		return false
	}
	for i, arg := range o.args {
		if o2.args[i] != arg {
			return false
		}
	}
	return true
}

func (o *op) write(w io.Writer) {
	fmt.Fprintf(w, "v%02d = %s", o.id, o.op)
	for _, arg := range o.args {
		fmt.Fprintf(w, " v%02d", arg.id)
	}
	if o.c != nil {
		fmt.Fprintf(w, " %v", o.c)
	}
	if o.name != "" {
		fmt.Fprintf(w, " %q", o.name)
	}
	if o.kind != nil {
		fmt.Fprintf(w, " [%s]", o.kind.typ)
	}
	fmt.Fprintf(w, "\n")
}

func (fn *Func) write(w io.Writer) {
	fmt.Fprintf(w, "FUNC %s\n", fn.name)
	for _, op := range fn.ops {
		op.write(w)
	}
}

func (f *File) Compile() {
	// TODO: CSE constants across the whole file

	fmt.Fprintf(f.w, `#include "go_asm.h"
#include "textflag.h"

`)

	for _, c := range f.consts {
		f.emitConst(c.name, c.data)
	}

	trace := func(fn *Func, step string) {
		if !logCompile {
			return
		}
		log.Printf("## Compiling %s: %s", fn.name, step)
		fn.write(os.Stderr)
	}

	for _, fn := range f.funcs {
		trace(fn, "initial")

		for {
			if fn.cse() {
				trace(fn, "post cse")
				continue
			}
			if fn.deadcode() {
				trace(fn, "post deadcode")
				continue
			}
			break
		}
		fn.addLoads()
		trace(fn, "post addLoads")

		// Assigning locations requires ops to be in dependency order.
		fn.schedule()
		trace(fn, "post schedule")

		locs := fn.assignLocs()

		fn.emit(f, locs)
	}
}

// cse performs common subexpression elimination.
func (fn *Func) cse() bool {
	// Compute structural hashes
	hashes := make(map[*op]uint64)
	var h maphash.Hash
	var bbuf [8]byte
	for _, op := range fn.ops {
		// We ignore the name for canonicalization.
		h.Reset()
		h.WriteString(op.op)
		// TODO: Ideally we would hash o1.c, but we don't have a good way to do that.
		for _, arg := range op.args {
			if _, ok := hashes[arg]; !ok {
				panic("ops not in dependency order")
			}
			binary.NativeEndian.PutUint64(bbuf[:], hashes[arg])
			h.Write(bbuf[:])
		}
		hashes[op] = h.Sum64()
	}

	canon := make(map[uint64][]*op)
	lookup := func(o *op) *op {
		hash := hashes[o]
		for _, o2 := range canon[hash] {
			if o.equalNoName(o2) {
				return o2
			}
		}
		canon[hash] = append(canon[hash], o)
		return o
	}

	// Canonicalize ops.
	dirty := false
	for _, op := range fn.ops {
		for i, arg := range op.args {
			newArg := lookup(arg)
			if arg != newArg {
				dirty = true
				op.args[i] = newArg
			}
		}
	}
	return dirty
}

// deadcode eliminates unused ops.
func (fn *Func) deadcode() bool {
	marks := make(map[*op]bool)
	var mark func(o *op)
	mark = func(o *op) {
		if marks[o] {
			return
		}
		marks[o] = true
		for _, arg := range o.args {
			mark(arg)
		}
	}
	// Mark operations that have a side-effect.
	for _, op := range fn.ops {
		switch op.op {
		case "return":
			mark(op)
		}
	}
	// Discard unmarked operations
	if len(marks) == len(fn.ops) {
		return false
	}
	newOps := make([]*op, 0, len(marks))
	for _, op := range fn.ops {
		if marks[op] {
			newOps = append(newOps, op)
		}
	}
	fn.ops = newOps
	return true
}

// canMem is a map from operation to a bitmap of which arguments can use a
// direct memory reference.
var canMem = map[string]uint64{
	"VPERMB":         1 << 0,
	"VPERMI2B":       1 << 0,
	"VPERMT2B":       1 << 0,
	"VGF2P8AFFINEQB": 1 << 0,
	"VPORQ":          1 << 0,
	"VPSUBQ":         1 << 0,
	"VPSHUFBITQMB":   1 << 0,
}

// addLoads inserts load ops for ops that can't take memory inputs directly.
func (fn *Func) addLoads() {
	// A lot of operations can directly take memory locations. If there's only a
	// single reference to a deref operation, and the operation can do the deref
	// itself, eliminate the deref. If there's more than one reference, then we
	// leave the load so we can share the value in the register.
	nRefs := fn.opRefs()
	loads := make(map[*op]*op) // deref -> load
	for _, o := range fn.ops {
		canMask := canMem[o.op]
		for i, arg := range o.args {
			// TODO: Many AVX-512 operations that support memory operands also
			// support a ".BCST" suffix that performs a broadcasting memory
			// load. If the const can be broadcast and all uses support
			// broadcast load, it would be nice to use .BCST. I'm not sure if
			// that belongs in this pass or a different one.
			if arg.op == "deref" || arg.op == "const" {
				// These produce memory locations.
				if canMask&(1<<i) == 0 || nRefs[arg] > 1 {
					// This argument needs to be loaded into a register.
					load, ok := loads[arg]
					if !ok {
						load = makeLoad(arg)
						fn.attach(load)
						loads[arg] = load
					}
					o.args[i] = load
				}
			}
		}
	}
}

func (fn *Func) opRefs() map[*op]int {
	refs := make(map[*op]int)
	for _, o1 := range fn.ops {
		for _, arg := range o1.args {
			refs[arg]++
		}
	}
	return refs
}

func makeLoad(deref *op) *op {
	var inst string
	switch deref.kind.reg {
	default:
		fatalf("don't know how to load %v", deref.kind.reg)
	case regClassGP:
		inst = "MOVQ"
	case regClassZ:
		inst = "VMOVDQU64"
	}
	// The load references deref rather than deref.args[0] because when we
	// assign locations, the deref op gets the memory location to load from,
	// while its argument has some other location (like a register). Also, the
	// offset to deref is attached to the deref op.
	return &op{op: inst, kind: deref.kind, args: []*op{deref}}
}

type opHeap []*op

func (h opHeap) Len() int      { return len(h) }
func (h opHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h opHeap) Less(i, j int) bool {
	priority := func(o *op) int {
		if o.op == "deref" || o.op == "const" {
			// Input to memory load
			return 1
		}
		if len(o.args) > 0 && (o.args[0].op == "deref" || o.args[0].op == "const") {
			// Memory load
			return 2
		}
		return 100
	}
	if p1, p2 := priority(h[i]), priority(h[j]); p1 != p2 {
		return p1 < p2
	}
	return h[i].id < h[j].id
}

func (h *opHeap) Push(x any) {
	*h = append(*h, x.(*op))
}

func (h *opHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

// schedule ensures fn's ops are in dependency order.
func (fn *Func) schedule() {
	// TODO: This tends to generate a huge amount of register pressure, mostly
	// because it floats loads as early as possible and partly because it has no
	// concept of rematerialization and CSE can make rematerializable values
	// live for a very long time. It some sense it doesn't matter because we
	// don't run out of registers for anything we need.

	missing := make(map[*op]int)
	uses := make(map[*op][]*op)
	var h opHeap
	for _, op := range fn.ops {
		if len(op.args) == 0 {
			h = append(h, op)
		} else {
			missing[op] = len(op.args)
		}
		for _, arg := range op.args {
			uses[arg] = append(uses[arg], op)
		}
	}
	heap.Init(&h)

	newOps := make([]*op, 0, len(fn.ops))
	for len(h) > 0 {
		if false {
			log.Printf("schedule: %s", h)
		}
		top := h[0]
		newOps = append(newOps, top)
		heap.Pop(&h)
		for _, o := range uses[top] {
			missing[o]--
			if missing[o] == 0 {
				heap.Push(&h, o)
			}
		}
	}
	if len(newOps) != len(fn.ops) {
		log.Print("schedule didn't schedule all ops")
		log.Print("before:")
		fn.write(os.Stderr)
		fn.ops = newOps
		log.Print("after:")
		fn.write(os.Stderr)
		log.Fatal("bad schedule")
	}

	fn.ops = newOps
}

func (fn *Func) emit(f *File, locs map[*op]loc) {
	w := f.w

	// Emit constants first
	for _, o := range fn.ops {
		if o.op == "const" {
			name := locs[o].(locMem).name
			f.emitConst(name, o.c)
		}
	}

	fmt.Fprintf(w, "TEXT %s(SB), NOSPLIT, $0-0\n", fn.name)

	// Emit body
	for _, o := range fn.ops {
		switch o.op {
		case "const", "arg", "return", "deref", "imm":
			// Does not produce code
			continue
		}
		switch o.op {
		case "addConst":
			fatalf("addConst not lowered")
		}

		opName := o.op
		// A ".mask" suffix is used to distinguish AVX-512 ops that use the same
		// mnemonic for regular and masked mode.
		opName = strings.TrimSuffix(opName, ".mask")

		fmt.Fprintf(w, "\t%s", opName)
		if o.op == "VGF2P8AFFINEQB" {
			// Hidden immediate, but always 0
			//
			// TODO: Replace this with an imm input.
			fmt.Fprintf(w, " $0,")
		}
		for i, arg := range o.args {
			if i == 0 {
				fmt.Fprintf(w, " ")
			} else {
				fmt.Fprintf(w, ", ")
			}
			if arg.op == "imm" {
				fmt.Fprintf(w, "$0x%x", arg.c)
			} else {
				fmt.Fprint(w, locs[arg].LocString())
			}
		}
		if _, ok := opRMW[o.op]; ok {
			// Read-modify-write instructions, so the output is already in the
			// arguments above.
		} else {
			fmt.Fprintf(w, ", %s", locs[o].LocString())
		}
		fmt.Fprintf(w, "\n")
	}
	fmt.Fprintf(w, "\tRET\n")
	fmt.Fprintf(w, "\n")
}

func (f *File) emitConst(name string, data any) {
	switch data := data.(type) {
	case []*Func:
		fmt.Fprintf(f.w, "GLOBL %s(SB), RODATA, $%#x\n", name, len(data)*8)
		for i, fn := range data {
			fmt.Fprintf(f.w, "DATA  %s+%#02x(SB)/8, ", name, 8*i)
			if fn == nil {
				fmt.Fprintf(f.w, "$0\n")
			} else {
				fmt.Fprintf(f.w, "$%s(SB)\n", fn.name)
			}
		}
		fmt.Fprintf(f.w, "\n")
		return
	}

	// Assume it's a numeric slice or array
	rv := reflect.ValueOf(data)
	sz := int(rv.Type().Elem().Size())
	fmt.Fprintf(f.w, "GLOBL %s(SB), RODATA, $%#x\n", name, rv.Len()*sz)
	for wi := 0; wi < sz*rv.Len()/8; wi++ { // Iterate over words
		var word uint64
		for j := 0; j < 8/sz; j++ { // Iterate over elements in this word
			d := rv.Index(wi*8/sz + j).Uint()
			word |= d << (j * sz * 8)
		}
		fmt.Fprintf(f.w, "DATA  %s+%#02x(SB)/8, $%#016x\n", name, 8*wi, word)
	}

	fmt.Fprintf(f.w, "\n")
}
