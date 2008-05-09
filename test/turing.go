// $G $F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// brainfuck

func main() {
       var a [30000]byte;
       prog := "++++++++++[>+++++++>++++++++++>+++>+<<<<-]>++.>+.+++++++..+++.>++.<<+++++++++++++++.>.+++.------.--------.>+.>.";
       p := 0;
       pc := 0;
       for {
               switch prog[pc] {
                       case '>':
                               p++;
                       case '<':
                               p--;
                       case '+':
                               a[p]++;
                       case '-':
                               a[p]--;
                       case '.':
                               print string(a[p]);
                       case '[':
                               if a[p] == 0 {
                                       for nest := 1; nest > 0; pc++ {
                                               if prog[pc+1] == ']' {
                                                       nest--;
                                               }
                                               if prog[pc+1] == '[' {
                                                       nest++;
                                               }
                                       }
                               }
                       case ']':
                               if a[p] != 0 {
                                       for nest := -1; nest < 0; pc-- {
                                               if prog[pc-1] == ']' {
                                                       nest--;
                                               }
                                               if prog[pc-1] == '[' {
                                                       nest++;
                                               }
                                       }
                               }
                       default:
                               return;
               }
               pc++;
       }
}
