// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test simulating a Turing machine, sort of.

package main

// brainfuck

var p, pc int
var a [30000]byte

const prog = "++++++++++[>+++++++>++++++++++>+++>+<<<<-]>++.>+.+++++++..+++.>++.<<+++++++++++++++.>.+++.------.--------.>+.>.!"

func scan(dir int) {
	for nest := dir; dir*nest > 0; pc += dir {
		switch prog[pc+dir] {
		case ']':
			nest--
		case '[':
			nest++
		}
	}
}

func main() {
	r := ""
	for {
		switch prog[pc] {
		case '>':
			p++
		case '<':
			p--
		case '+':
			a[p]++
		case '-':
			a[p]--
		case '.':
			r += string(a[p])
		case '[':
			if a[p] == 0 {
				scan(1)
			}
		case ']':
			if a[p] != 0 {
				scan(-1)
			}
		default:
			if r != "Hello World!\n" {
				panic(r)
			}
			return
		}
		pc++
	}
}
