// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "fmt"

func printFunc(f *Func) {
	fmt.Print(f.Name)
	fmt.Print(" ")
	fmt.Println(f.Type)
	printed := make([]bool, f.NumValues())
	for _, b := range f.Blocks {
		fmt.Printf("  b%d:\n", b.ID)
		n := 0

		// print phis first since all value cycles contain a phi
		for _, v := range b.Values {
			if v.Op != OpPhi {
				continue
			}
			fmt.Print("    ")
			fmt.Println(v.LongString())
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
				fmt.Print("    ")
				fmt.Println(v.LongString())
				printed[v.ID] = true
				n++
			}
			if m == n {
				fmt.Println("dependency cycle!")
				for _, v := range b.Values {
					if printed[v.ID] {
						continue
					}
					fmt.Print("    ")
					fmt.Println(v.LongString())
					printed[v.ID] = true
					n++
				}
			}
		}

		fmt.Println("    " + b.LongString())
	}
}
