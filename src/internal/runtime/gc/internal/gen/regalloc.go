// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gen

import (
	"fmt"
	"log"
	"math/bits"
	"strings"
)

const traceRegAlloc = true

type regClass uint8

const (
	regClassFixed regClass = iota
	regClassGP
	regClassZ
	regClassK

	numRegClasses

	regClassNone = ^regClass(0)
)

type locReg struct {
	cls regClass
	reg int
}

func (l locReg) LocString() string {
	switch l.cls {
	case regClassFixed:
		return fixedRegs[l.reg]
	case regClassGP:
		return gpRegs[l.reg]
	case regClassZ:
		return fmt.Sprintf("Z%d", l.reg)
	case regClassK:
		return fmt.Sprintf("K%d", l.reg)
	}
	panic("bad register class")
}

func (l locReg) Deref(off int) (loc, error) {
	return locMem{l, off, ""}, nil
}

func (l locReg) Reg() (locReg, bool) {
	return l, true
}

type locMem struct {
	base locReg
	off  int
	name string
}

func (l locMem) LocString() string {
	if l.base.cls == regClassFixed && l.base.reg == regSB && l.off == 0 {
		return l.name + "(SB)"
	}
	if l.name != "" {
		return fmt.Sprintf("%s+%d(%s)", l.name, l.off, l.base.LocString())
	}
	if l.off != 0 {
		return fmt.Sprintf("%d(%s)", l.off, l.base.LocString())
	}
	return "(" + l.base.LocString() + ")"
}

func (l locMem) Deref(off int) (loc, error) {
	return nil, fmt.Errorf("cannot dereference already memory address %s", l.LocString())
}

func (l locMem) Reg() (locReg, bool) {
	if l.base.cls == regClassFixed {
		return locReg{}, false
	}
	return l.base, true
}

type loc interface {
	LocString() string          // Return the assembly syntax for this location
	Deref(off int) (loc, error) // Treat this location as an address and return a location with the contents of memory at that address
	Reg() (locReg, bool)        // Register used by this location
}

var opRMW = map[string]int{
	"VPERMI2B":          2, // Overwrites third argument
	"VPERMI2B.Z":        3, // Overwrites fourth argument
	"VPERMI2B.mask":     3, // Overwrites fourth argument
	"VPERMT2B":          1, // Overwrites second argument TODO: Check this. Unused for now.
	"VPBROADCASTQ.mask": 2, // Overwrites last argument
}

// TODO: Should we have a general rule that all ".mask" instructions overwrite
// their last argument?

const (
	regSB = iota
	regFP
)

var fixedRegs = []string{regSB: "SB", regFP: "FP"}
var gpRegs = []string{"AX", "BX", "CX", "DI", "SI", "R8", "R9", "R10", "R11"} // ABI argument order

type regSet struct {
	inUse [numRegClasses]uint32
}

func (s *regSet) used(o *op, l loc) {
	if l == nil {
		return
	}
	reg, ok := l.Reg()
	if !ok {
		return
	}
	if traceRegAlloc {
		log.Printf("  alloc %s @ v%02d", reg.LocString(), o.id)
	}
	if s.inUse[reg.cls]&(1<<reg.reg) != 0 {
		fatalf("register %s already used", reg.LocString())
	}
	s.inUse[reg.cls] |= 1 << reg.reg
}

func (s *regSet) free(l loc) {
	if l == nil {
		return
	}
	reg, ok := l.Reg()
	if !ok {
		return
	}
	if traceRegAlloc {
		log.Printf("  free %s", reg.LocString())
	}
	if s.inUse[reg.cls]&(1<<reg.reg) == 0 {
		fatalf("register %s is not in use", reg.LocString())
	}
	s.inUse[reg.cls] &^= 1 << reg.reg
}

func (fn *Func) assignLocs() map[*op]loc {
	// Remove static indicator on name, if any. We'll add it back.
	nameBase := strings.TrimSuffix(fn.name, "<>")

	// Create map from op -> fn.ops index
	opIndexes := make(map[*op]int, len(fn.ops))
	for i, o := range fn.ops {
		opIndexes[o] = i
	}

	// Read-modify-write operations share a location with one of their inputs.
	// Likewise, deref ops extend the lifetime of their input (but in a shared
	// way, unlike RMW ops).
	//
	// Compute a map from each op to the earliest "canonical" op whose live
	// range we'll use.
	canon := make(map[*op]*op)
	overwritten := make(map[*op]bool)
	for _, o := range fn.ops {
		// Check that this op doesn't use any overwritten inputs.
		for _, arg := range o.args {
			if overwritten[arg] {
				// TODO: The solution to this is to insert copy ops.
				fatalf("op %+v uses overwritten input %+v", o, arg)
			}
		}

		// Record canonical op.
		rmw, ok := opRMW[o.op]
		if ok {
			canon[o] = canon[o.args[rmw]]
			// Record that the input is dead now and must not be referenced.
			overwritten[o.args[rmw]] = true
		} else if o.op == "deref" {
			canon[o] = canon[o.args[0]]
		} else {
			canon[o] = o
		}
	}

	// Compute live ranges of each canonical op.
	//
	// First, find the last use of each op.
	lastUses := make(map[*op]*op) // Canonical creation op -> last use op
	for _, op := range fn.ops {
		for _, arg := range op.args {
			lastUses[canon[arg]] = op
		}
	}
	// Invert the last uses map to get a map from op to the (canonical) values
	// that die at that op.
	lastUseMap := make(map[*op][]*op) // op of last use -> (canonical) creation ops
	for def, lastUse := range lastUses {
		lastUseMap[lastUse] = append(lastUseMap[lastUse], def)
	}

	// Prepare for assignments
	regUsed := make([]regSet, len(fn.ops)) // In-use registers at each op
	for i := range regUsed {
		// X15/Y15/Z15 is reserved by the Go ABI
		regUsed[i].inUse[regClassZ] |= 1 << 15
		// K0 is contextual (if used as an opmask, it means no mask). Too
		// complicated, so just ignore it.
		regUsed[i].inUse[regClassK] |= 1 << 0
	}
	locs := make(map[*op]loc)
	assign := func(o *op, l loc) {
		if have, ok := locs[o]; ok {
			fatalf("op %+v already assigned location %v (new %v)", o, have, l)
			return
		}
		if o == canon[o] {
			// Mark this location used over o's live range
			for i := opIndexes[o]; i < opIndexes[lastUses[o]]; i++ {
				regUsed[i].used(fn.ops[i], l)
			}
		}
		locs[o] = l
	}

	// Assign fixed locations
	id := 0
	for _, o := range fn.ops {
		switch o.op {
		case "arg":
			if traceRegAlloc {
				log.Printf("fixed op %+v", o)
			}
			assign(o, o.c.(locReg))
		case "const":
			if traceRegAlloc {
				log.Printf("fixed op %+v", o)
			}
			name := o.name
			if name == "" {
				name = fmt.Sprintf("%s_%d<>", nameBase, id)
				id++
			} else if name[0] == '*' {
				name = nameBase + name[1:]
			}
			assign(o, locMem{locReg{cls: regClassFixed, reg: regSB}, 0, name})
		case "return":
			if traceRegAlloc {
				log.Printf("fixed op %+v", o)
			}
			assign(o, nil) // no location
			// TODO: argZ should start at 0.
			argGP, argZ := 0, 1
			for _, arg := range o.args {
				switch arg.kind.reg {
				default:
					fatalf("bad register class for return value")
				case regClassGP:
					assign(canon[arg], locReg{regClassGP, argGP})
					argGP++
				case regClassZ:
					assign(canon[arg], locReg{regClassZ, argZ})
					argZ++
				}
			}
		case "imm":
			assign(o, nil) // no location
		}
	}

	// Assign locations.
	for _, o := range fn.ops {
		if traceRegAlloc {
			log.Printf("assign %+v", o)
		}

		if _, ok := locs[o]; ok {
			// Already assigned a fixed location above.
			continue
		}

		if o.op == "deref" {
			loc, err := locs[o.args[0]].Deref(o.c.(int))
			if err != nil {
				fatalf("%v", err)
			}
			// We don't "assign" here because we've already processed the
			// canonical op, which marked loc's register as in-use for the whole
			// live range.
			locs[o] = loc
			continue
		}

		if canon[o] != o {
			// Copy the canonical op's location.
			locs[o] = locs[canon[o]]
			continue
		}
		// Below here we know that o is already a canonical op.

		if _, ok := opRMW[o.op]; ok {
			fatalf("read-modify-write op not canonicalized")
		}

		// Find a free register of the right class.
		cls := o.kind.reg
		var used uint32
		for i := opIndexes[o]; i < opIndexes[lastUses[o]]; i++ {
			used |= regUsed[i].inUse[cls]
		}

		// Assign result location.
		num := bits.TrailingZeros32(^used)
		switch cls {
		default:
			fatalf("unknown reg class %v", cls)
		case regClassGP:
			if num >= len(gpRegs) {
				panic("out of GP regs")
			}
		case regClassZ:
			if num >= 32 {
				panic("out of Z regs")
			}
		case regClassK:
			if num >= 8 {
				panic("out of K regs")
			}
		}
		loc := locReg{cls, num}
		assign(o, loc)
	}

	return locs
}
