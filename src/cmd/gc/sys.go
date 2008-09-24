// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.


package foop	// rename to avoid redeclaration

func	mal(uint32) *any;
func	breakpoint();
func	throwindex();
func	throwreturn();
func	panicl(int32);

func	printbool(bool);
func	printfloat(double);
func	printint(int64);
func	printstring(string);
func	printpointer(*any);
func	printinter(any);

func	catstring(string, string) string;
func	cmpstring(string, string) int32;
func	slicestring(string, int32, int32) string;
func	indexstring(string, int32) byte;
func	intstring(int64) string;
func	byteastring(*byte, int32) string;
func	arraystring(*[]byte) string;

func	ifaceT2I(sigi *byte, sigt *byte, elem any) (ret any);
func	ifaceI2T(sigt *byte, iface any) (ret any);
func	ifaceI2I(sigi *byte, iface any) (ret any);

func	argc() int32;
func	envc() int32;
func	argv(int32) string;
func	envv(int32) string;

func	frexp(float64) (float64, int32);	// break fp into exp,fract
func	ldexp(float64, int32) float64;		// make fp from exp,fract
func	modf(float64) (float64, float64);	// break fp into double.double
func	isInf(float64, int32) bool;		// test for infinity
func	isNaN(float64) bool;			// test for not-a-number
func	Inf(int32) float64;			// return signed Inf
func	NaN() float64;				// return a NaN

func	newmap(keysize uint32, valsize uint32,
		keyalg uint32, valalg uint32,
		hint uint32) (hmap *map[any]any);
func	mapaccess1(hmap *map[any]any, key any) (val any);
func	mapaccess2(hmap *map[any]any, key any) (val any, pres bool);
func	mapassign1(hmap *map[any]any, key any, val any);
func	mapassign2(hmap *map[any]any, key any, val any, pres bool);

func	newchan(elemsize uint32, elemalg uint32, hint uint32) (hchan *chan any);
func	chanrecv1(hchan *chan any) (elem any);
func	chanrecv2(hchan *chan any) (elem any, pres bool);
func	chanrecv3(hchan *chan any, elem *any) (pres bool);
func	chansend1(hchan *chan any, elem any);
func	chansend2(hchan *chan any, elem any) (pres bool);

func	newselect(size uint32) (sel *byte);
func	selectsend(sel *byte, hchan *chan any, elem any) (selected bool);
func	selectrecv(sel *byte, hchan *chan any, elem *any) (selected bool);
func	selectgo(sel *byte);

func	newarray(nel uint32, cap uint32, width uint32) (ary *[]any);
func	arraysliced(old *[]any, lb uint32, hb uint32, width uint32) (ary *[]any);
func	arrayslices(old *any, nel uint32, lb uint32, hb uint32, width uint32) (ary *[]any);
func	arrays2d(old *any, nel uint32) (ary *[]any);

func	gosched();
func	goexit();

func	readfile(string) (string, bool);	// read file into string; boolean status
func	writefile(string, string) (bool);	// write string into file; boolean status
func	bytestorune(*byte, int32, int32) (int32, int32);	// convert bytes to runes	
func	stringtorune(string, int32, int32) (int32, int32);	// convert bytes to runes	

func	exit(int32);

export
	mal
	breakpoint
	throwindex
	throwreturn

	// print panic
	panicl
	printbool
	printfloat
	printint
	printstring
	printpointer
	printinter

	// op string
	catstring
	cmpstring
	slicestring
	indexstring
	intstring
	byteastring
	arraystring

	// interface
	ifaceT2I
	ifaceI2T
	ifaceI2I

	// args
	argc
	envc
	argv
	envv

	// fp
	frexp
	ldexp
	modf
	isInf,
	isNaN,
	Inf,
	NaN,

	// map
	newmap
	mapaccess1
	mapaccess2
	mapassign1
	mapassign2

	// chan
	newchan
	chanrecv1
	chanrecv2
	chanrecv3
	chansend1
	chansend2

	// select
	newselect
	selectsend
	selectrecv
	selectgo

	// dynamic arrays
	newarray
	arraysliced
	arrayslices
	arrays2d

	// go routines
	gosched
	goexit

	// files
	readfile
	writefile

	// runes and utf-8
	bytestorune
	stringtorune

	// system calls
	exit
	;
