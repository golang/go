// $G $D/$F.go && $L $F.$A && ./$A.out || echo BUG: fails incorrectly

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type S struct {
}

func (p *S) M() {
}

type I interface {
	M();
}

func main() {
	var p *S = nil;
	var i I = p;  // this should be possible even though p is nil: we still know the type
	i.M();  // should be possible since we know the type, and don't ever use the receiver
}


/*
throw: ifaces2i: nil pointer
SIGSEGV: segmentation violation
Faulting address: 0x0
pc: 0x1b7d

0x1b7d?zi
	throw(30409, 0, 0, ...)
	throw(0x76c9, 0x0, 0x0, ...)
0x207f?zi
	sys·ifaces2i(31440, 0, 31480, ...)
	sys·ifaces2i(0x7ad0, 0x7af8, 0x0, ...)
0x136f?zi
	main·main(1, 0, 1606416424, ...)
	main·main(0x1, 0x7fff5fbff828, 0x0, ...)

rax     0x1
rbx     0x1
rcx     0x33b5
rdx     0x0
rdi     0x1
rsi     0x7684
rbp     0x7684
rsp     0xafb8
r8      0x0
r9      0x0
r10     0x1002
r11     0x206
r12     0x0
r13     0x0
r14     0x7c48
r15     0xa000
rip     0x1b7d
rflags  0x10202
cs      0x27
fs      0x10
gs      0x48
*/
