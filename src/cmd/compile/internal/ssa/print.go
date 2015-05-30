// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"bytes"
	"fmt"
	"io"
	"os"
)

func printFunc(f *Func) {
	fprintFunc(os.Stdout, f)
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
		fmt.Fprintf(w, "  b%d:\n", b.ID)
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
					if w.Block == b && !printed[w.ID] {
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
