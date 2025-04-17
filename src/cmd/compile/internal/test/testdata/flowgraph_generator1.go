// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"strings"
)

// make fake flow graph.

// The blocks of the flow graph are designated with letters A
// through Z, always including A (start block) and Z (exit
// block) The specification of a flow graph is a comma-
// separated list of block successor words, for blocks ordered
// A, B, C etc, where each block except Z has one or two
// successors, and any block except A can be a target. Within
// the generated code, each block with two successors includes
// a conditional testing x & 1 != 0 (x is the input parameter
// to the generated function) and also unconditionally shifts x
// right by one, so that different inputs generate different
// execution paths, including loops. Every block inverts a
// global binary to ensure it is not empty. For a flow graph
// with J words (J+1 blocks), a J-1 bit serial number specifies
// which blocks (not including A and Z) include an increment of
// the return variable y by increasing powers of 10, and a
// different version of the test function is created for each
// of the 2-to-the-(J-1) serial numbers.

// For each generated function a compact summary is also
// created so that the generated function can be simulated
// with a simple interpreter to sanity check the behavior of
// the compiled code.

// For example:

// func BC_CD_BE_BZ_CZ101(x int64) int64 {
// 	y := int64(0)
// 	var b int64
// 	_ = b
// 	b = x & 1
// 	x = x >> 1
// 	if b != 0 {
// 		goto C
// 	}
// 	goto B
// B:
// 	glob_ = !glob_
// 	y += 1
// 	b = x & 1
// 	x = x >> 1
// 	if b != 0 {
// 		goto D
// 	}
// 	goto C
// C:
// 	glob_ = !glob_
// 	// no y increment
// 	b = x & 1
// 	x = x >> 1
// 	if b != 0 {
// 		goto E
// 	}
// 	goto B
// D:
// 	glob_ = !glob_
// 	y += 10
// 	b = x & 1
// 	x = x >> 1
// 	if b != 0 {
// 		goto Z
// 	}
// 	goto B
// E:
// 	glob_ = !glob_
// 	// no y increment
// 	b = x & 1
// 	x = x >> 1
// 	if b != 0 {
// 		goto Z
// 	}
// 	goto C
// Z:
// 	return y
// }

// {f:BC_CD_BE_BZ_CZ101,
//  maxin:32, blocks:[]blo{
//  	blo{inc:0, cond:true, succs:[2]int64{1, 2}},
//  	blo{inc:1, cond:true, succs:[2]int64{2, 3}},
//  	blo{inc:0, cond:true, succs:[2]int64{1, 4}},
//  	blo{inc:10, cond:true, succs:[2]int64{1, 25}},
//  	blo{inc:0, cond:true, succs:[2]int64{2, 25}},}},

var labels string = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

func blocks(spec string) (blocks []string, fnameBase string) {
	spec = strings.ToUpper(spec)
	blocks = strings.Split(spec, ",")
	fnameBase = strings.ReplaceAll(spec, ",", "_")
	return
}

func makeFunctionFromFlowGraph(blocks []blo, fname string) string {
	s := ""

	for j := range blocks {
		// begin block
		if j == 0 {
			// block A, implicit label
			s += `
func ` + fname + `(x int64) int64 {
	y := int64(0)
	var b int64
	_ = b`
		} else {
			// block B,C, etc, explicit label w/ conditional increment
			l := labels[j : j+1]
			yeq := `
	// no y increment`
			if blocks[j].inc != 0 {
				yeq = `
	y += ` + fmt.Sprintf("%d", blocks[j].inc)
			}

			s += `
` + l + `:
	glob = !glob` + yeq
		}

		// edges to successors
		if blocks[j].cond { // conditionally branch to second successor
			s += `
	b = x & 1
	x = x >> 1
	if b != 0 {` + `
		goto ` + string(labels[blocks[j].succs[1]]) + `
	}`

		}
		// branch to first successor
		s += `
	goto ` + string(labels[blocks[j].succs[0]])
	}

	// end block (Z)
	s += `
Z:
	return y
}
`
	return s
}

var graphs []string = []string{
	"Z", "BZ,Z", "B,BZ", "BZ,BZ",
	"ZB,Z", "B,ZB", "ZB,BZ", "ZB,ZB",

	"BC,C,Z", "BC,BC,Z", "BC,BC,BZ",
	"BC,Z,Z", "BC,ZC,Z", "BC,ZC,BZ",
	"BZ,C,Z", "BZ,BC,Z", "BZ,CZ,Z",
	"BZ,C,BZ", "BZ,BC,BZ", "BZ,CZ,BZ",
	"BZ,C,CZ", "BZ,BC,CZ", "BZ,CZ,CZ",

	"BC,CD,BE,BZ,CZ",
	"BC,BD,CE,CZ,BZ",
	"BC,BD,CE,FZ,GZ,F,G",
	"BC,BD,CE,FZ,GZ,G,F",

	"BC,DE,BE,FZ,FZ,Z",
	"BC,DE,BE,FZ,ZF,Z",
	"BC,DE,BE,ZF,FZ,Z",
	"BC,DE,EB,FZ,FZ,Z",
	"BC,ED,BE,FZ,FZ,Z",
	"CB,DE,BE,FZ,FZ,Z",

	"CB,ED,BE,FZ,FZ,Z",
	"BC,ED,EB,FZ,ZF,Z",
	"CB,DE,EB,ZF,FZ,Z",
	"CB,ED,EB,FZ,FZ,Z",

	"BZ,CD,CD,CE,BZ",
	"EC,DF,FG,ZC,GB,BE,FD",
	"BH,CF,DG,HE,BF,CG,DH,BZ",
}

// blo describes a block in the generated/interpreted code
type blo struct {
	inc   int64 // increment amount
	cond  bool  // block ends in conditional
	succs [2]int64
}

// strings2blocks converts a slice of strings specifying
// successors into a slice of blo encoding the blocks in a
// common form easy to execute or interpret.
func strings2blocks(blocks []string, fname string, i int) (bs []blo, cond uint) {
	bs = make([]blo, len(blocks))
	edge := int64(1)
	cond = 0
	k := uint(0)
	for j, s := range blocks {
		if j == 0 {
		} else {
			if (i>>k)&1 != 0 {
				bs[j].inc = edge
				edge *= 10
			}
			k++
		}
		if len(s) > 1 {
			bs[j].succs[1] = int64(blocks[j][1] - 'A')
			bs[j].cond = true
			cond++
		}
		bs[j].succs[0] = int64(blocks[j][0] - 'A')
	}
	return bs, cond
}

// fmtBlocks writes out the blocks for consumption in the generated test
func fmtBlocks(bs []blo) string {
	s := "[]blo{"
	for _, b := range bs {
		s += fmt.Sprintf("blo{inc:%d, cond:%v, succs:[2]int64{%d, %d}},", b.inc, b.cond, b.succs[0], b.succs[1])
	}
	s += "}"
	return s
}

func main() {
	fmt.Printf(`// This is a machine-generated test file from flowgraph_generator1.go.
package main
import "fmt"
var glob bool
`)
	s := "var funs []fun = []fun{"
	for _, g := range graphs {
		split, fnameBase := blocks(g)
		nconfigs := 1 << uint(len(split)-1)

		for i := 0; i < nconfigs; i++ {
			fname := fnameBase + fmt.Sprintf("%b", i)
			bs, k := strings2blocks(split, fname, i)
			fmt.Printf("%s", makeFunctionFromFlowGraph(bs, fname))
			s += `
		{f:` + fname + `, maxin:` + fmt.Sprintf("%d", 1<<k) + `, blocks:` + fmtBlocks(bs) + `},`
		}

	}
	s += `}
`
	// write types for name+array tables.
	fmt.Printf("%s",
		`
type blo struct {
	inc   int64
	cond  bool
	succs [2]int64
}
type fun struct {
	f      func(int64) int64
	maxin  int64
	blocks []blo
}
`)
	// write table of function names and blo arrays.
	fmt.Printf("%s", s)

	// write interpreter and main/test
	fmt.Printf("%s", `
func interpret(blocks []blo, x int64) (int64, bool) {
	y := int64(0)
	last := int64(25) // 'Z'-'A'
	j := int64(0)
	for i := 0; i < 4*len(blocks); i++ {
		b := blocks[j]
		y += b.inc
		next := b.succs[0]
		if b.cond {
			c := x&1 != 0
			x = x>>1
			if c {
				next = b.succs[1]
			}
		}
		if next == last {
			return y, true
		}
		j = next
	}
	return -1, false
}

func main() {
	sum := int64(0)
	for i, f := range funs {
		for x := int64(0); x < 16*f.maxin; x++ {
			y, ok := interpret(f.blocks, x)
			if ok {
				yy := f.f(x)
				if y != yy {
					fmt.Printf("y(%d) != yy(%d), x=%b, i=%d, blocks=%v\n", y, yy, x, i, f.blocks)
					return
				}
				sum += y
			}
		}
	}
//	fmt.Printf("Sum of all returns over all terminating inputs is %d\n", sum)
}
`)
}
