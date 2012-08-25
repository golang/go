// cmpout

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "go/ast";

func g(list []ast.Expr) {
	n := len(list)-1;
	println(list[n].Pos());
}


// f is the same as g except that the expression assigned to n is inlined.
func f(list []ast.Expr) {
	// n := len(list)-1;
	println(list[len(list)-1 /* n */].Pos());
}


func main() {
	list := []ast.Expr{&ast.Ident{}};
	g(list);  // this works
	f(list);  // this doesn't
}


/*
0
throw: index out of range

panic PC=0x2bcf10
throw+0x33 /home/gri/go/src/pkg/runtime/runtime.c:71
	throw(0x470f8, 0x0)
sys·throwindex+0x1c /home/gri/go/src/pkg/runtime/runtime.c:45
	sys·throwindex()
main·f+0x26 /home/gri/go/test/bugs/bug206.go:16
	main·f(0x2b9560, 0x0)
main·main+0xc3 /home/gri/go/test/bugs/bug206.go:23
	main·main()
mainstart+0xf /home/gri/go/src/pkg/runtime/amd64/asm.s:55
	mainstart()
goexit /home/gri/go/src/pkg/runtime/proc.c:133
	goexit()
*/
