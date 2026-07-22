// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacore

import (
	"fmt"
	"io"
	"strings"

	"cmd/compile/internal/ssa/ssaop"
	"cmd/internal/hash"
	"cmd/internal/src"
)

func FprintFunc(p funcPrinter, f *Func) {
	reachable, live := findlive(f)
	defer f.Cache.FreeBoolSlice(live)
	p.Header(f)
	printed := make([]bool, f.NumValues())
	for _, b := range f.Blocks {
		p.StartBlock(b, reachable[b.ID])

		if f.Scheduled {
			// Order of Values has been decided - print in that order.
			for _, v := range b.Values {
				p.Value(v, live[v.ID])
				printed[v.ID] = true
			}
			p.EndBlock(b, reachable[b.ID])
			continue
		}

		// print phis first since all value cycles contain a phi
		n := 0
		for _, v := range b.Values {
			if v.Op != ssaop.OpPhi {
				continue
			}
			p.Value(v, live[v.ID])
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
				p.Value(v, live[v.ID])
				printed[v.ID] = true
				n++
			}
			if m == n {
				p.StartDepCycle()
				for _, v := range b.Values {
					if printed[v.ID] {
						continue
					}
					p.Value(v, live[v.ID])
					printed[v.ID] = true
					n++
				}
				p.EndDepCycle()
			}
		}

		p.EndBlock(b, reachable[b.ID])
	}
	for _, name := range f.Names {
		p.Named(name, f.NamedValues[name])
	}
}

func PrintFunc(f *Func) {
	f.Logf("%s", f)
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

type StringFuncPrinter struct {
	w         io.Writer
	printDead bool
}

type funcPrinter interface {
	Header(f *Func)
	StartBlock(b *Block, reachable bool)
	EndBlock(b *Block, reachable bool)
	Value(v *Value, live bool)
	StartDepCycle()
	EndDepCycle()
	Named(n LocalSlot, vals []*Value)
}

func HashFunc(f *Func) []byte {
	h := hash.New32()
	p := StringFuncPrinter{w: h, printDead: true}
	FprintFunc(p, f)
	return h.Sum(nil)
}

func (f *Func) String() string {
	var buf strings.Builder
	p := StringFuncPrinter{w: &buf, printDead: true}
	FprintFunc(p, f)
	return buf.String()
}

// RewriteHash returns a hash of f suitable for detecting rewrite cycles.
func (f *Func) RewriteHash() string {
	h := hash.New32()
	p := StringFuncPrinter{w: h, printDead: false}
	FprintFunc(p, f)
	return fmt.Sprintf("%x", h.Sum(nil))
}

func (p StringFuncPrinter) Header(f *Func) {
	fmt.Fprint(p.w, f.Name)
	fmt.Fprint(p.w, " ")
	fmt.Fprintln(p.w, f.Type)
}

func (p StringFuncPrinter) StartBlock(b *Block, reachable bool) {
	if !p.printDead && !reachable {
		return
	}
	fmt.Fprintf(p.w, "  b%d:", b.ID)
	if len(b.Preds) > 0 {
		io.WriteString(p.w, " <-")
		for _, e := range b.Preds {
			pred := e.B
			fmt.Fprintf(p.w, " b%d", pred.ID)
		}
	}
	if !reachable {
		fmt.Fprint(p.w, " DEAD")
	}
	io.WriteString(p.w, "\n")
}

func (p StringFuncPrinter) EndBlock(b *Block, reachable bool) {
	if !p.printDead && !reachable {
		return
	}
	fmt.Fprintln(p.w, "    "+b.LongString())
}

func (p StringFuncPrinter) Value(v *Value, live bool) {
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

func (p StringFuncPrinter) StartDepCycle() {
	fmt.Fprintln(p.w, "dependency cycle!")
}

func (p StringFuncPrinter) EndDepCycle() {}

func (p StringFuncPrinter) Named(n LocalSlot, vals []*Value) {
	fmt.Fprintf(p.w, "name %s: %v\n", n, vals)
}
