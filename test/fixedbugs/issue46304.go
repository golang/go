// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This testcase caused a crash when the register ABI was in effect,
// on amd64 (problem with register allocation).

package main

type Op struct {
	tag   string
	_x    []string
	_q    [20]uint64
	plist []P
}

type P struct {
	tag string
	_x  [10]uint64
	b   bool
}

type M int

//go:noinline
func (w *M) walkP(p *P) *P {
	np := &P{}
	*np = *p
	np.tag += "new"
	return np
}

func (w *M) walkOp(op *Op) *Op {
	if op == nil {
		return nil
	}

	orig := op
	cloned := false
	clone := func() {
		if !cloned {
			cloned = true
			op = &Op{}
			*op = *orig
		}
	}

	pCloned := false
	for i := range op.plist {
		if s := w.walkP(&op.plist[i]); s != &op.plist[i] {
			if !pCloned {
				pCloned = true
				clone()
				op.plist = make([]P, len(orig.plist))
				copy(op.plist, orig.plist)
			}
			op.plist[i] = *s
		}
	}

	return op
}

func main() {
	var ww M
	w := &ww
	p1 := P{tag: "a"}
	p1._x[1] = 9
	o := Op{tag: "old", plist: []P{p1}}
	no := w.walkOp(&o)
	if no.plist[0].tag != "anew" {
		panic("bad")
	}
}
