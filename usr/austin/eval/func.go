// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eval

import (
	"container/vector";
	"eval";
)

/*
 * Virtual machine
 */

type vm struct {
	pc uint;
	// The current execution frame.  If execution is within a
	// block, this may be a child of the original function
	// activation frame.
	f *Frame;
	// The original function activation frame.  This is used to
	// access function out args.
	activation *Frame;
}

type code []func(*vm)

func (i code) exec(fr *Frame) {
	v := vm{0, fr, fr};

	l := uint(len(i));
	for v.pc < l {
		pc := v.pc;
		v.pc++;
		i[pc](&v);
	}
}

/*
 * Code buffer
 */

type codeBuf struct {
	instrs code;
}

func newCodeBuf() *codeBuf {
	return &codeBuf{make(code, 0, 16)};
}

func (b *codeBuf) push(instr func(*vm)) {
	n := len(b.instrs);
	if n >= cap(b.instrs) {
		a := make(code, n, n*2);
		for i := range b.instrs {
			a[i] = b.instrs[i];
		}
		b.instrs = a;
	}
	b.instrs = b.instrs[0:n+1];
	b.instrs[n] = instr;
}

func (b *codeBuf) get() code {
	// Freeze this buffer into an array of exactly the right size
	a := make(code, len(b.instrs));
	for i := range b.instrs {
		a[i] = b.instrs[i];
	}
	return code(a);
}

/*
 * User-defined functions
 */

type evalFunc struct {
	sc *Scope;
	fr *Frame;
	code code;
}

func (f *evalFunc) NewFrame() *Frame {
	return f.sc.NewFrame(f.fr);
}

func (f *evalFunc) Call(fr *Frame) {
	f.code.exec(fr);
}
