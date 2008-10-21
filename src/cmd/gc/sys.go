// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.


package SYS	// rename to avoid redeclaration

export func	mal(uint32) *any;
export func	breakpoint();
export func	throwindex();
export func	throwreturn();
export func	panicl(int32);

export func	printbool(bool);
export func	printfloat(double);
export func	printint(int64);
export func	printstring(string);
export func	printpointer(*any);
export func	printinter(any);
export func	printnl();
export func	printsp();

export func	catstring(string, string) string;
export func	cmpstring(string, string) int32;
export func	slicestring(string, int32, int32) string;
export func	indexstring(string, int32) byte;
export func	intstring(int64) string;
export func	byteastring(*byte, int32) string;
export func	arraystring(*[]byte) string;

export func	ifaceT2I(sigi *byte, sigt *byte, elem any) (ret any);
export func	ifaceI2T(sigt *byte, iface any) (ret any);
export func	ifaceI2I(sigi *byte, iface any) (ret any);
export func	ifaceeq(i1 any, i2 any) (ret bool);
export func	reflect(i interface { }) (uint64, string);

export func	argc() int32;
export func	envc() int32;
export func	argv(int32) string;
export func	envv(int32) string;

export func	frexp(float64) (float64, int32);	// break fp into exp,fract
export func	ldexp(float64, int32) float64;		// make fp from exp,fract
export func	modf(float64) (float64, float64);	// break fp into double.double
export func	isInf(float64, int32) bool;		// test for infinity
export func	isNaN(float64) bool;			// test for not-a-number
export func	Inf(int32) float64;			// return signed Inf
export func	NaN() float64;				// return a NaN

export func	newmap(keysize uint32, valsize uint32,
			keyalg uint32, valalg uint32,
			hint uint32) (hmap *map[any]any);
export func	mapaccess1(hmap *map[any]any, key any) (val any);
export func	mapaccess2(hmap *map[any]any, key any) (val any, pres bool);
export func	mapassign1(hmap *map[any]any, key any, val any);
export func	mapassign2(hmap *map[any]any, key any, val any, pres bool);

export func	newchan(elemsize uint32, elemalg uint32, hint uint32) (hchan *chan any);
export func	chanrecv1(hchan *chan any) (elem any);
export func	chanrecv2(hchan *chan any) (elem any, pres bool);
export func	chanrecv3(hchan *chan any, elem *any) (pres bool);
export func	chansend1(hchan *chan any, elem any);
export func	chansend2(hchan *chan any, elem any) (pres bool);

export func	newselect(size uint32) (sel *byte);
export func	selectsend(sel *byte, hchan *chan any, elem any) (selected bool);
export func	selectrecv(sel *byte, hchan *chan any, elem *any) (selected bool);
export func	selectgo(sel *byte);

export func	newarray(nel uint32, cap uint32, width uint32) (ary *[]any);
export func	arraysliced(old *[]any, lb uint32, hb uint32, width uint32) (ary *[]any);
export func	arrayslices(old *any, nel uint32, lb uint32, hb uint32, width uint32) (ary *[]any);
export func	arrays2d(old *any, nel uint32) (ary *[]any);

export func	gosched();
export func	goexit();

export func	readfile(string) (string, bool);	// read file into string; boolean status
export func	writefile(string, string) (bool);	// write string into file; boolean status
export func	bytestorune(*byte, int32, int32) (int32, int32);	// convert bytes to runes
export func	stringtorune(string, int32) (int32, int32);	// convert bytes to runes

export func	exit(int32);
