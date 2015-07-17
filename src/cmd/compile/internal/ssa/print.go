// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"bytes"
	"fmt"
	"io"
)

func printFunc(f *Func) {
	f.Logf("%s", f)
}

func (f *Func) String() string {
	var buf bytes.Buffer
	fprintFunc(&buf, f)
	return buf.String()
}

func fprintFunc(w io.Writer, f *Func) {
	fmt.Fprint(w, f.Name)
	fmt.Fprint(w, " ")
	fmt.Fprintln(w, f.Type)
	printed := make([]bool, f.NumValues())
	for _, b := range f.Blocks {
		fmt.Fprintf(w, "  b%d:", b.ID)
		if len(b.Preds) > 0 {
			io.WriteString(w, " <-")
			for _, pred := range b.Preds {
				fmt.Fprintf(w, " b%d", pred.ID)
			}
		}
		io.WriteString(w, "\n")
		n := 0

		// print phis first since all value cycles contain a phi
		for _, v := range b.Values {
			if v.Op != OpPhi {
				continue
			}
			fmt.Fprint(w, "    ")
			fmt.Fprintln(w, v.LongString())
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
				fmt.Fprint(w, "    ")
				fmt.Fprintln(w, v.LongString())
				printed[v.ID] = true
				n++
			}
			if m == n {
				fmt.Fprintln(w, "dependency cycle!")
				for _, v := range b.Values {
					if printed[v.ID] {
						continue
					}
					fmt.Fprint(w, "    ")
					fmt.Fprintln(w, v.LongString())
					printed[v.ID] = true
					n++
				}
			}
		}

		fmt.Fprintln(w, "    "+b.LongString())
	}
}
