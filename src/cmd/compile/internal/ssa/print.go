// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"fmt"
	"io"
	"strings"

	"cmd/internal/notsha256"
	"cmd/internal/src"
)

func printFunc(f *Func) {
	f.Logf("%s", f)
}

func hashFunc(f *Func) []byte {
	h := notsha256.New()
	p := stringFuncPrinter{w: h, printDead: true}
	fprintFunc(p, f)
	return h.Sum(nil)
}

func (f *Func) String() string {
	var buf strings.Builder
	p := stringFuncPrinter{w: &buf, printDead: true}
	fprintFunc(p, f)
	return buf.String()
}

// rewriteHash returns a hash of f suitable for detecting rewrite cycles.
func (f *Func) rewriteHash() string {
	h := notsha256.New()
	p := stringFuncPrinter{w: h, printDead: false}
	fprintFunc(p, f)
	return fmt.Sprintf("%x", h.Sum(nil))
}

type funcPrinter interface {
	header(f *Func)
	startBlock(b *Block, reachable bool)
	endBlock(b *Block, reachable bool)
	value(v *Value, live bool)
	startDepCycle()
	endDepCycle()
	named(n LocalSlot, vals []*Value)
}

type stringFuncPrinter struct {
	w         io.Writer
	printDead bool
}

func (p stringFuncPrinter) header(f *Func) {
	fmt.Fprint(p.w, f.Name)
	fmt.Fprint(p.w, " ")
	fmt.Fprintln(p.w, f.Type)
}

func (p stringFuncPrinter) startBlock(b *Block, reachable bool) {
	if !p.printDead && !reachable {
		return
	}
	fmt.Fprintf(p.w, "  b%d:", b.ID)
	if len(b.Preds) > 0 {
		io.WriteString(p.w, " <-")
		for _, e := range b.Preds {
			pred := e.b
			fmt.Fprintf(p.w, " b%d", pred.ID)
		}
	}
	if !reachable {
		fmt.Fprint(p.w, " DEAD")
	}
	io.WriteString(p.w, "\n")
}

func (p stringFuncPrinter) endBlock(b *Block, reachable bool) {
	if !p.printDead && !reachable {
		return
	}
	fmt.Fprintln(p.w, "    "+b.LongString())
}

func StmtString(p src.XPos) string {
	linenumber := "(?) "
	if p.IsKnown() {
		pfx := ""
		if p.IsStmt() == src.PosIsStmt {
			pfx = "+"
		}
		if p.IsStmt() == src.PosNotStmt {
			pfx = "-"
		}
		linenumber = fmt.Sprintf("(%s%d) ", pfx, p.Line())
	}
	return linenumber
}

func (p stringFuncPrinter) value(v *Value, live bool) {
	if !p.printDead && !live {
		return
	}
	fmt.Fprintf(p.w, "    %s", StmtString(v.Pos))
	fmt.Fprint(p.w, v.LongString())
	if !live {
		fmt.Fprint(p.w, " DEAD")
	}
	fmt.Fprintln(p.w)
}

func (p stringFuncPrinter) startDepCycle() {
	fmt.Fprintln(p.w, "dependency cycle!")
}

func (p stringFuncPrinter) endDepCycle() {}

func (p stringFuncPrinter) named(n LocalSlot, vals []*Value) {
	fmt.Fprintf(p.w, "name %s: %v\n", n, vals)
}

func fprintFunc(p funcPrinter, f *Func) {
	reachable, live := findlive(f)
	defer f.retDeadcodeLive(live)
	p.header(f)
	printed := make([]bool, f.NumValues())
	for _, b := range f.Blocks {
		p.startBlock(b, reachable[b.ID])

		if f.scheduled {
			// Order of Values has been decided - print in that order.
			for _, v := range b.Values {
				p.value(v, live[v.ID])
				printed[v.ID] = true
			}
			p.endBlock(b, reachable[b.ID])
			continue
		}

		// print phis first since all value cycles contain a phi
		n := 0
		for _, v := range b.Values {
			if v.Op != OpPhi {
				continue
			}
			p.value(v, live[v.ID])
			printed[v.ID] = true
			n++
		}

		// print rest of values in dependency order
		for n < len(b.Values) {
			m := n
		outer:
			for _, v := range b.Values {
				if printed[v.ID] {
					continue
				}
				for _, w := range v.Args {
					// w == nil shouldn't happen, but if it does,
					// don't panic; we'll get a better diagnosis later.
					if w != nil && w.Block == b && !printed[w.ID] {
						continue outer
					}
				}
				p.value(v, live[v.ID])
				printed[v.ID] = true
				n++
			}
			if m == n {
				p.startDepCycle()
				for _, v := range b.Values {
					if printed[v.ID] {
						continue
					}
					p.value(v, live[v.ID])
					printed[v.ID] = true
					n++
				}
				p.endDepCycle()
			}
		}

		p.endBlock(b, reachable[b.ID])
	}
	for _, name := range f.Names {
		p.named(*name, f.NamedValues[*name])
	}
}
