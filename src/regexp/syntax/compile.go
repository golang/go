// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import "unicode"

// A patchList is a list of instruction pointers that need to be filled in (patched).
// Because the pointers haven't been filled in yet, we can reuse their storage
// to hold the list. It's kind of sleazy, but works well in practice.
// See https://swtch.com/~rsc/regexp/regexp1.html for inspiration.
//
// These aren't really pointers: they're integers, so we can reinterpret them
// this way without using package unsafe. A value l.head denotes
// p.inst[l.head>>1].Out (l.head&1==0) or .Arg (l.head&1==1).
// head == 0 denotes the empty list, okay because we start every program
// with a fail instruction, so we'll never want to point at its output link.
type patchList struct {
	head, tail uint32
}

func makePatchList(n uint32) patchList {
	return patchList{n, n}
}

func (l patchList) patch(p *Prog, val uint32) {
	head := l.head
	for head != 0 {
		i := &p.Inst[head>>1]
		if head&1 == 0 {
			head = i.Out
			i.Out = val
		} else {
			head = i.Arg
			i.Arg = val
		}
	}
}

func (l1 patchList) append(p *Prog, l2 patchList) patchList {
	if l1.head == 0 {
		return l2
	}
	if l2.head == 0 {
		return l1
	}

	i := &p.Inst[l1.tail>>1]
	if l1.tail&1 == 0 {
		i.Out = l2.head
	} else {
		i.Arg = l2.head
	}
	return patchList{l1.head, l2.tail}
}

// A frag represents a compiled program fragment.
type frag struct {
	i   uint32    // index of first instruction
	out patchList // where to record end instruction
}

type compiler struct {
	p *Prog
}

// Compile compiles the regexp into a program to be executed.
// The regexp should have been simplified already (returned from re.Simplify).
func Compile(re *Regexp) (*Prog, error) {
	var c compiler
	c.init()
	f := c.compile(re)
	f.out.patch(c.p, c.inst(InstMatch).i)
	c.p.Start = int(f.i)
	return c.p, nil
}

func (c *compiler) init() {
	c.p = new(Prog)
	c.p.NumCap = 2 // implicit ( and ) for whole match $0
	c.inst(InstFail)
}

var anyRuneNotNL = []rune{0, '\n' - 1, '\n' + 1, unicode.MaxRune}
var anyRune = []rune{0, unicode.MaxRune}

func (c *compiler) compile(re *Regexp) frag {
	switch re.Op {
	case OpNoMatch:
		return c.fail()
	case OpEmptyMatch:
		return c.nop()
	case OpLiteral:
		if len(re.Rune) == 0 {
			return c.nop()
		}
		var f frag
		for j := range re.Rune {
			f1 := c.rune(re.Rune[j:j+1], re.Flags)
			if j == 0 {
				f = f1
			} else {
				f = c.cat(f, f1)
			}
		}
		return f
	case OpCharClass:
		return c.rune(re.Rune, re.Flags)
	case OpAnyCharNotNL:
		return c.rune(anyRuneNotNL, 0)
	case OpAnyChar:
		return c.rune(anyRune, 0)
	case OpBeginLine:
		return c.empty(EmptyBeginLine)
	case OpEndLine:
		return c.empty(EmptyEndLine)
	case OpBeginText:
		return c.empty(EmptyBeginText)
	case OpEndText:
		return c.empty(EmptyEndText)
	case OpWordBoundary:
		return c.empty(EmptyWordBoundary)
	case OpNoWordBoundary:
		return c.empty(EmptyNoWordBoundary)
	case OpCapture:
		bra := c.cap(uint32(re.Cap << 1))
		sub := c.compile(re.Sub[0])
		ket := c.cap(uint32(re.Cap<<1 | 1))
		return c.cat(c.cat(bra, sub), ket)
	case OpStar:
		return c.star(c.compile(re.Sub[0]), re.Flags&NonGreedy != 0)
	case OpPlus:
		return c.plus(c.compile(re.Sub[0]), re.Flags&NonGreedy != 0)
	case OpQuest:
		return c.quest(c.compile(re.Sub[0]), re.Flags&NonGreedy != 0)
	case OpConcat:
		if len(re.Sub) == 0 {
			return c.nop()
		}
		var f frag
		for i, sub := range re.Sub {
			if i == 0 {
				f = c.compile(sub)
			} else {
				f = c.cat(f, c.compile(sub))
			}
		}
		return f
	case OpAlternate:
		var f frag
		for _, sub := range re.Sub {
			f = c.alt(f, c.compile(sub))
		}
		return f
	}
	panic("regexp: unhandled case in compile")
}

func (c *compiler) inst(op InstOp) frag {
	// TODO: impose length limit
	f := frag{i: uint32(len(c.p.Inst))}
	c.p.Inst = append(c.p.Inst, Inst{Op: op})
	return f
}

func (c *compiler) nop() frag {
	f := c.inst(InstNop)
	f.out = makePatchList(f.i << 1)
	return f
}

func (c *compiler) fail() frag {
	return frag{}
}

func (c *compiler) cap(arg uint32) frag {
	f := c.inst(InstCapture)
	f.out = makePatchList(f.i << 1)
	c.p.Inst[f.i].Arg = arg

	if c.p.NumCap < int(arg)+1 {
		c.p.NumCap = int(arg) + 1
	}
	return f
}

func (c *compiler) cat(f1, f2 frag) frag {
	// concat of failure is failure
	if f1.i == 0 || f2.i == 0 {
		return frag{}
	}

	// TODO: elide nop

	f1.out.patch(c.p, f2.i)
	return frag{f1.i, f2.out}
}

func (c *compiler) alt(f1, f2 frag) frag {
	// alt of failure is other
	if f1.i == 0 {
		return f2
	}
	if f2.i == 0 {
		return f1
	}

	f := c.inst(InstAlt)
	i := &c.p.Inst[f.i]
	i.Out = f1.i
	i.Arg = f2.i
	f.out = f1.out.append(c.p, f2.out)
	return f
}

func (c *compiler) quest(f1 frag, nongreedy bool) frag {
	f := c.inst(InstAlt)
	i := &c.p.Inst[f.i]
	if nongreedy {
		i.Arg = f1.i
		f.out = makePatchList(f.i << 1)
	} else {
		i.Out = f1.i
		f.out = makePatchList(f.i<<1 | 1)
	}
	f.out = f.out.append(c.p, f1.out)
	return f
}

func (c *compiler) star(f1 frag, nongreedy bool) frag {
	f := c.inst(InstAlt)
	i := &c.p.Inst[f.i]
	if nongreedy {
		i.Arg = f1.i
		f.out = makePatchList(f.i << 1)
	} else {
		i.Out = f1.i
		f.out = makePatchList(f.i<<1 | 1)
	}
	f1.out.patch(c.p, f.i)
	return f
}

func (c *compiler) plus(f1 frag, nongreedy bool) frag {
	return frag{f1.i, c.star(f1, nongreedy).out}
}

func (c *compiler) empty(op EmptyOp) frag {
	f := c.inst(InstEmptyWidth)
	c.p.Inst[f.i].Arg = uint32(op)
	f.out = makePatchList(f.i << 1)
	return f
}

func (c *compiler) rune(r []rune, flags Flags) frag {
	f := c.inst(InstRune)
	i := &c.p.Inst[f.i]
	i.Rune = r
	flags &= FoldCase // only relevant flag is FoldCase
	if len(r) != 1 || unicode.SimpleFold(r[0]) == r[0] {
		// and sometimes not even that
		flags &^= FoldCase
	}
	i.Arg = uint32(flags)
	f.out = makePatchList(f.i << 1)

	// Special cases for exec machine.
	switch {
	case flags&FoldCase == 0 && (len(r) == 1 || len(r) == 2 && r[0] == r[1]):
		i.Op = InstRune1
	case len(r) == 2 && r[0] == 0 && r[1] == unicode.MaxRune:
		i.Op = InstRuneAny
	case len(r) == 4 && r[0] == 0 && r[1] == '\n'-1 && r[2] == '\n'+1 && r[3] == unicode.MaxRune:
		i.Op = InstRuneAnyNotNL
	}

	return f
}
