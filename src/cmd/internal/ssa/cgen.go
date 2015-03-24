// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "fmt"

// cgen selects machine instructions for the function.
// This pass generates assembly output for now, but should
// TODO(khr): generate binary output (via liblink?) instead of text.
func cgen(f *Func) {
	fmt.Printf("TEXT %s(SB),0,$0\n", f.Name) // TODO: frame size / arg size

	// TODO: prolog, allocate stack frame

	// hack for now, until regalloc is done
	f.RegAlloc = make([]Location, f.NumValues())

	for idx, b := range f.Blocks {
		fmt.Printf("%d:\n", b.ID)
		for _, v := range b.Values {
			asm := opcodeTable[v.Op].asm
			fmt.Print("\t")
			if asm == "" {
				fmt.Print("\t")
			}
			for i := 0; i < len(asm); i++ {
				switch asm[i] {
				default:
					fmt.Printf("%c", asm[i])
				case '%':
					i++
					switch asm[i] {
					case '%':
						fmt.Print("%")
					case 'I':
						i++
						n := asm[i] - '0'
						if f.RegAlloc[v.Args[n].ID] != nil {
							fmt.Print(f.RegAlloc[v.Args[n].ID].Name())
						} else {
							fmt.Printf("v%d", v.Args[n].ID)
						}
					case 'O':
						i++
						n := asm[i] - '0'
						if n != 0 {
							panic("can only handle 1 output for now")
						}
						if f.RegAlloc[v.ID] != nil {
							// TODO: output tuple
							fmt.Print(f.RegAlloc[v.ID].Name())
						} else {
							fmt.Printf("v%d", v.ID)
						}
					case 'A':
						fmt.Print(v.Aux)
					}
				}
			}
			fmt.Println("\t; " + v.LongString())
		}
		// find next block in layout sequence
		var next *Block
		if idx < len(f.Blocks)-1 {
			next = f.Blocks[idx+1]
		}
		// emit end of block code
		// TODO: this is machine specific
		switch b.Kind {
		case BlockPlain:
			if b.Succs[0] != next {
				fmt.Printf("\tJMP\t%d\n", b.Succs[0].ID)
			}
		case BlockExit:
			// TODO: run defers (if any)
			// TODO: deallocate frame
			fmt.Println("\tRET")
		case BlockCall:
			// nothing to emit - call instruction already happened
		case BlockEQ:
			if b.Succs[0] == next {
				fmt.Printf("\tJNE\t%d\n", b.Succs[1].ID)
			} else if b.Succs[1] == next {
				fmt.Printf("\tJEQ\t%d\n", b.Succs[0].ID)
			} else {
				fmt.Printf("\tJEQ\t%d\n", b.Succs[0].ID)
				fmt.Printf("\tJMP\t%d\n", b.Succs[1].ID)
			}
		case BlockNE:
			if b.Succs[0] == next {
				fmt.Printf("\tJEQ\t%d\n", b.Succs[1].ID)
			} else if b.Succs[1] == next {
				fmt.Printf("\tJNE\t%d\n", b.Succs[0].ID)
			} else {
				fmt.Printf("\tJNE\t%d\n", b.Succs[0].ID)
				fmt.Printf("\tJMP\t%d\n", b.Succs[1].ID)
			}
		case BlockLT:
			if b.Succs[0] == next {
				fmt.Printf("\tJGE\t%d\n", b.Succs[1].ID)
			} else if b.Succs[1] == next {
				fmt.Printf("\tJLT\t%d\n", b.Succs[0].ID)
			} else {
				fmt.Printf("\tJLT\t%d\n", b.Succs[0].ID)
				fmt.Printf("\tJMP\t%d\n", b.Succs[1].ID)
			}
		default:
			fmt.Printf("\t%s ->", b.Kind.String())
			for _, s := range b.Succs {
				fmt.Printf(" %d", s.ID)
			}
			fmt.Printf("\n")
		}
	}
}
