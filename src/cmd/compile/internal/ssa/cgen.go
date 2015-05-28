// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"bytes"
	"fmt"
	"os"
)

// cgen selects machine instructions for the function.
// This pass generates assembly output for now, but should
// TODO(khr): generate binary output (via liblink?) instead of text.
func cgen(f *Func) {
	fmt.Printf("TEXT %s(SB),0,$0\n", f.Name) // TODO: frame size / arg size

	// TODO: prolog, allocate stack frame

	for idx, b := range f.Blocks {
		fmt.Printf("%d:\n", b.ID)
		for _, v := range b.Values {
			var buf bytes.Buffer
			asm := opcodeTable[v.Op].asm
			buf.WriteString("        ")
			for i := 0; i < len(asm); i++ {
				switch asm[i] {
				default:
					buf.WriteByte(asm[i])
				case '\t':
					buf.WriteByte(' ')
					for buf.Len()%8 != 0 {
						buf.WriteByte(' ')
					}
				case '%':
					i++
					switch asm[i] {
					case '%':
						buf.WriteByte('%')
					case 'I':
						i++
						n := asm[i] - '0'
						if f.RegAlloc[v.Args[n].ID] != nil {
							buf.WriteString(f.RegAlloc[v.Args[n].ID].Name())
						} else {
							fmt.Fprintf(&buf, "v%d", v.Args[n].ID)
						}
					case 'O':
						i++
						n := asm[i] - '0'
						if n != 0 {
							panic("can only handle 1 output for now")
						}
						if f.RegAlloc[v.ID] != nil {
							buf.WriteString(f.RegAlloc[v.ID].Name())
						} else {
							fmt.Fprintf(&buf, "v%d", v.ID)
						}
					case 'A':
						fmt.Fprint(&buf, v.Aux)
					}
				}
			}
			for buf.Len() < 40 {
				buf.WriteByte(' ')
			}
			buf.WriteString("; ")
			buf.WriteString(v.LongString())
			buf.WriteByte('\n')
			os.Stdout.Write(buf.Bytes())
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
		case BlockULT:
			if b.Succs[0] == next {
				fmt.Printf("\tJAE\t%d\n", b.Succs[1].ID)
			} else if b.Succs[1] == next {
				fmt.Printf("\tJB\t%d\n", b.Succs[0].ID)
			} else {
				fmt.Printf("\tJB\t%d\n", b.Succs[0].ID)
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
