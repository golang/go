// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.


package PACKAGE

// emitted by compiler, not referred to by go programs

func	mal(int32) *any;
func	throwindex();
func	throwreturn();
func	panicl(int32);

func	printbool(bool);
func	printfloat(float64);
func	printint(int64);
func	printstring(string);
func	printpointer(any);
func	printinter(any);
func	printarray(any);
func	printnl();
func	printsp();

func	catstring(string, string) string;
func	cmpstring(string, string) int;
func	slicestring(string, int, int) string;
func	indexstring(string, int) byte;
func	intstring(int64) string;
func	byteastring(*byte, int) string;
func	arraystring([]byte) string;
func	stringiter(string, int) int;
func	stringiter2(string, int) (retk int, retv int);

func	ifaceT2I(sigi *byte, sigt *byte, elem any) (ret any);
func	ifaceI2T(sigt *byte, iface any) (ret any);
func	ifaceI2T2(sigt *byte, iface any) (ret any, ok bool);
func	ifaceI2I(sigi *byte, iface any) (ret any);
func	ifaceI2I2(sigi *byte, iface any) (ret any, ok bool);
func	ifaceeq(i1 any, i2 any) (ret bool);
func	ifacethash(i1 any) (ret uint32);

func	newmap(keysize int, valsize int,
			keyalg int, valalg int,
			hint int) (hmap map[any]any);
func	mapaccess1(hmap map[any]any, key any) (val any);
func	mapaccess2(hmap map[any]any, key any) (val any, pres bool);
func	mapassign1(hmap map[any]any, key any, val any);
func	mapassign2(hmap map[any]any, key any, val any, pres bool);
func	mapiterinit(hmap map[any]any, hiter *any);
func	mapiternext(hiter *any);
func	mapiter1(hiter *any) (key any);
func	mapiter2(hiter *any) (key any, val any);

func	newchan(elemsize int, elemalg int, hint int) (hchan chan any);
func	chanrecv1(hchan chan any) (elem any);
func	chanrecv2(hchan chan any) (elem any, pres bool);
func	chanrecv3(hchan chan any, elem *any) (pres bool);
func	chansend1(hchan chan any, elem any);
func	chansend2(hchan chan any, elem any) (pres bool);
func	closechan(hchan chan any);
func	closedchan(hchan chan any) bool;

func	newselect(size int) (sel *byte);
func	selectsend(sel *byte, hchan chan any, elem any) (selected bool);
func	selectrecv(sel *byte, hchan chan any, elem *any) (selected bool);
func	selectdefault(sel *byte) (selected bool);
func	selectgo(sel *byte);

func	newarray(nel int, cap int, width int) (ary []any);
func	arraysliced(old []any, lb int, hb int, width int) (ary []any);
func	arrayslices(old *any, nel int, lb int, hb int, width int) (ary []any);
func	arrays2d(old *any, nel int) (ary []any);

func	closure();	// has args, but compiler fills in

// used by go programs

func	Breakpoint();

func	Reflect(i interface { }) (uint64, string, bool);
func	Unreflect(uint64, string, bool) (ret interface { });

var	Args []string;
var	Envs []string;

func	Gosched();
func	Goexit();

func	Exit(int);

func	Caller(n int) (pc uint64, file string, line int, ok bool);
