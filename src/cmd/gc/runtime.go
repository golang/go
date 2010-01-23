// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// NOTE: If you change this file you must run "./mkbuiltin"
// to update builtin.c.boot.  This is not done automatically
// to avoid depending on having a working compiler binary.

package PACKAGE

// emitted by compiler, not referred to by go programs

func mal(int32) *any
func throwindex()
func throwreturn()
func throwinit()
func panicl()

func printbool(bool)
func printfloat(float64)
func printint(int64)
func printuint(uint64)
func printstring(string)
func printpointer(any)
func printiface(any)
func printeface(any)
func printslice(any)
func printnl()
func printsp()

func catstring(string, string) string
func cmpstring(string, string) int
func slicestring(string, int, int) string
func slicestring1(string, int) string
func indexstring(string, int) byte
func intstring(int64) string
func slicebytetostring([]byte) string
func sliceinttostring([]int) string
func stringiter(string, int) int
func stringiter2(string, int) (retk int, retv int)
func slicecopy(to any, fr any, wid uint32) int

func ifaceI2E(iface any) (ret any)
func ifaceE2I(typ *byte, iface any) (ret any)
func ifaceT2E(typ *byte, elem any) (ret any)
func ifaceE2T(typ *byte, elem any) (ret any)
func ifaceE2I2(typ *byte, iface any) (ret any, ok bool)
func ifaceE2T2(typ *byte, elem any) (ret any, ok bool)
func ifaceT2I(typ1 *byte, typ2 *byte, elem any) (ret any)
func ifaceI2T(typ *byte, iface any) (ret any)
func ifaceI2T2(typ *byte, iface any) (ret any, ok bool)
func ifaceI2I(typ *byte, iface any) (ret any)
func ifaceI2Ix(typ *byte, iface any) (ret any)
func ifaceI2I2(typ *byte, iface any) (ret any, ok bool)
func ifaceeq(i1 any, i2 any) (ret bool)
func efaceeq(i1 any, i2 any) (ret bool)
func ifacethash(i1 any) (ret uint32)
func efacethash(i1 any) (ret uint32)

// *byte is really *runtime.Type
func makemap(key, val *byte, hint int) (hmap map[any]any)
func mapaccess1(hmap map[any]any, key any) (val any)
func mapaccess2(hmap map[any]any, key any) (val any, pres bool)
func mapassign1(hmap map[any]any, key any, val any)
func mapassign2(hmap map[any]any, key any, val any, pres bool)
func mapiterinit(hmap map[any]any, hiter *any)
func mapiternext(hiter *any)
func mapiter1(hiter *any) (key any)
func mapiter2(hiter *any) (key any, val any)

// *byte is really *runtime.Type
func makechan(elem *byte, hint int) (hchan chan any)
func chanrecv1(hchan <-chan any) (elem any)
func chanrecv2(hchan <-chan any) (elem any, pres bool)
func chansend1(hchan chan<- any, elem any)
func chansend2(hchan chan<- any, elem any) (pres bool)
func closechan(hchan any)
func closedchan(hchan any) bool

func newselect(size int) (sel *byte)
func selectsend(sel *byte, hchan chan<- any, elem any) (selected bool)
func selectrecv(sel *byte, hchan <-chan any, elem *any) (selected bool)
func selectdefault(sel *byte) (selected bool)
func selectgo(sel *byte)

func makeslice(typ *byte, nel int, cap int) (ary []any)
func sliceslice1(old []any, lb int, width int) (ary []any)
func sliceslice(old []any, lb int, hb int, width int) (ary []any)
func slicearray(old *any, nel int, lb int, hb int, width int) (ary []any)

func closure() // has args, but compiler fills in

// only used on 32-bit
func int64div(int64, int64) int64
func uint64div(uint64, uint64) uint64
func int64mod(int64, int64) int64
func uint64mod(uint64, uint64) uint64
func float64toint64(float64) int64
func int64tofloat64(int64) float64
