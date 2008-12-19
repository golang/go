// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.


package PACKAGE

export func	mal(int32) *any;
export func	breakpoint();
export func	throwindex();
export func	throwreturn();
export func	panicl(int32);

export func	printbool(bool);
export func	printfloat(float64);
export func	printint(int64);
export func	printstring(string);
export func	printpointer(*any);
export func	printinter(any);
export func	printarray(any);
export func	printnl();
export func	printsp();

export func	catstring(string, string) string;
export func	cmpstring(string, string) int;
export func	slicestring(string, int, int) string;
export func	indexstring(string, int) byte;
export func	intstring(int64) string;
export func	byteastring(*byte, int) string;
export func	arraystring([]byte) string;

export func	ifaceT2I(sigi *byte, sigt *byte, elem any) (ret any);
export func	ifaceI2T(sigt *byte, iface any) (ret any);
export func	ifaceI2T2(sigt *byte, iface any) (ret any, ok bool);
export func	ifaceI2I(sigi *byte, iface any) (ret any);
export func	ifaceI2I2(sigi *byte, iface any) (ret any, ok bool);
export func	ifaceeq(i1 any, i2 any) (ret bool);
export func	reflect(i interface { }) (uint64, string);
export func	unreflect(uint64, string) (ret interface { });

export func	argc() int;
export func	envc() int;
export func	argv(int) string;
export func	envv(int) string;

export func	frexp(float64) (float64, int);		// break fp into exp,fract
export func	ldexp(float64, int) float64;		// make fp from exp,fract
export func	modf(float64) (float64, float64);	// break fp into double.double
export func	isInf(float64, int) bool;		// test for infinity
export func	isNaN(float64) bool;			// test for not-a-number
export func	Inf(int) float64;			// return signed Inf
export func	NaN() float64;				// return a NaN
export func	float32bits(float32) uint32;		// raw bits
export func	float64bits(float64) uint64;		// raw bits
export func	float32frombits(uint32) float32;	// raw bits
export func	float64frombits(uint64) float64;	// raw bits

export func	newmap(keysize int, valsize int,
			keyalg int, valalg int,
			hint int) (hmap *map[any]any);
export func	mapaccess1(hmap *map[any]any, key any) (val any);
export func	mapaccess2(hmap *map[any]any, key any) (val any, pres bool);
export func	mapassign1(hmap *map[any]any, key any, val any);
export func	mapassign2(hmap *map[any]any, key any, val any, pres bool);
export func	mapiterinit(hmap *map[any]any, hiter *any);
export func	mapiternext(hiter *any);
export func	mapiter1(hiter *any) (key any);
export func	mapiter2(hiter *any) (key any, val any);

export func	newchan(elemsize int, elemalg int, hint int) (hchan *chan any);
export func	chanrecv1(hchan *chan any) (elem any);
export func	chanrecv2(hchan *chan any) (elem any, pres bool);
export func	chanrecv3(hchan *chan any, elem *any) (pres bool);
export func	chansend1(hchan *chan any, elem any);
export func	chansend2(hchan *chan any, elem any) (pres bool);

export func	newselect(size int) (sel *byte);
export func	selectsend(sel *byte, hchan *chan any, elem any) (selected bool);
export func	selectrecv(sel *byte, hchan *chan any, elem *any) (selected bool);
export func	selectdefault(sel *byte) (selected bool);
export func	selectgo(sel *byte);

export func	newarray(nel int, cap int, width int) (ary []any);
export func	arraysliced(old []any, lb int, hb int, width int) (ary []any);
export func	arrayslices(old *any, nel int, lb int, hb int, width int) (ary []any);
export func	arrays2d(old *any, nel int) (ary []any);

export func	gosched();
export func	goexit();

export func	readfile(string) (string, bool);	// read file into string; boolean status
export func	writefile(string, string) (bool);	// write string into file; boolean status
export func	bytestorune(*byte, int, int) (int, int);	// convert bytes to runes
export func	stringtorune(string, int) (int, int);	// convert bytes to runes

export func	exit(int);

export func	symdat() (symtab []byte, pclntab []byte);

export func	semacquire(sema *int32);
export func	semrelease(sema *int32);
