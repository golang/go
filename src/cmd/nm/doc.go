// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Nm lists the symbols defined or used by an object file, archive, or executable.
//
// Usage:
//	go tool nm [options] file...
//
// The default output prints one line per symbol, with three space-separated
// fields giving the address (in hexadecimal), type (a character), and name of
// the symbol. The types are:
//
//	T	text (code) segment symbol
//	t	static text segment symbol
//	R	read-only data segment symbol
//	r	static read-only data segment symbol
//	D	data segment symbol
//	d	static data segment symbol
//	B	bss segment symbol
//	b	static bss segment symbol
//	C	constant address
//	U	referenced but undefined symbol
//
// Following established convention, the address is omitted for undefined
// symbols (type U).
//
// The options control the printed output:
//
//	-n
//		an alias for -sort address (numeric),
//		for compatibility with other nm commands
//	-size
//		print symbol size in decimal between address and type
//	-sort {address,name,none,size}
//		sort output in the given order (default name)
//		size orders from largest to smallest
//	-type
//		print symbol type after name
//
package main
