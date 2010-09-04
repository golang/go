// $G $F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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
					print(string(a[p]))
			case '[':
				if a[p] == 0 {
					scan(1)
				}
			case ']':
				if a[p] != 0 {
					scan(-1)
				}
			default:
					return
		}
		pc++
	}
}
