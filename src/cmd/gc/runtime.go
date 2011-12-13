// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// NOTE: If you change this file you must run "./mkbuiltin"
// to update builtin.c.boot.  This is not done automatically
// to avoid depending on having a working compiler binary.

package PACKAGE

import "unsafe"

// emitted by compiler, not referred to by go programs

func new(typ *byte) *any
func panicindex()
func panicslice()
func throwreturn()
func throwinit()
func panicwrap(string, string, string)

func panic(interface{})
func recover(*int32) interface{}

func printbool(bool)
func printfloat(float64)
func printint(int64)
func printuint(uint64)
func printcomplex(complex128)
func printstring(string)
func printpointer(any)
func printiface(any)
func printeface(any)
func printslice(any)
func printnl()
func printsp()
func goprintf()

// filled in by compiler: int n, string, string, ...
func concatstring()

// filled in by compiler: Type*, int n, Slice, ...
func append()
func appendslice(typ *byte, x any, y []any) any
func appendstr(typ *byte, x []byte, y string) []byte

func cmpstring(string, string) int
func slicestring(string, int, int) string
func slicestring1(string, int) string
func intstring(int64) string
func slicebytetostring([]byte) string
func slicerunetostring([]rune) string
func stringtoslicebyte(string) []byte
func stringtoslicerune(string) []rune
func stringiter(string, int) int
func stringiter2(string, int) (retk int, retv rune)
func copy(to any, fr any, wid uint32) int
func slicestringcopy(to any, fr any) int

// interface conversions
func convI2E(elem any) (ret any)
func convI2I(typ *byte, elem any) (ret any)
func convT2E(typ *byte, elem any) (ret any)
func convT2I(typ *byte, typ2 *byte, elem any) (ret any)

// interface type assertions  x.(T)
func assertE2E(typ *byte, iface any) (ret any)
func assertE2E2(typ *byte, iface any) (ret any, ok bool)
func assertE2I(typ *byte, iface any) (ret any)
func assertE2I2(typ *byte, iface any) (ret any, ok bool)
func assertE2T(typ *byte, iface any) (ret any)
func assertE2T2(typ *byte, iface any) (ret any, ok bool)
func assertI2E(typ *byte, iface any) (ret any)
func assertI2E2(typ *byte, iface any) (ret any, ok bool)
func assertI2I(typ *byte, iface any) (ret any)
func assertI2I2(typ *byte, iface any) (ret any, ok bool)
func assertI2T(typ *byte, iface any) (ret any)
func assertI2T2(typ *byte, iface any) (ret any, ok bool)

func ifaceeq(i1 any, i2 any) (ret bool)
func efaceeq(i1 any, i2 any) (ret bool)
func ifacethash(i1 any) (ret uint32)
func efacethash(i1 any) (ret uint32)

func equal(typ *byte, x1, x2 any) (ret bool)

// *byte is really *runtime.Type
func makemap(mapType *byte, hint int64) (hmap map[any]any)
func mapaccess1(mapType *byte, hmap map[any]any, key any) (val any)
func mapaccess2(mapType *byte, hmap map[any]any, key any) (val any, pres bool)
func mapassign1(mapType *byte, hmap map[any]any, key any, val any)
func mapassign2(mapType *byte, hmap map[any]any, key any, val any, pres bool)
func mapiterinit(mapType *byte, hmap map[any]any, hiter *any)
func mapdelete(mapType *byte, hmap map[any]any, key any)
func mapiternext(hiter *any)
func mapiter1(hiter *any) (key any)
func mapiter2(hiter *any) (key any, val any)

// *byte is really *runtime.Type
func makechan(chanType *byte, hint int64) (hchan chan any)
func chanrecv1(chanType *byte, hchan <-chan any) (elem any)
func chanrecv2(chanType *byte, hchan <-chan any) (elem any, received bool)
func chansend1(chanType *byte, hchan chan<- any, elem any)
func closechan(hchan any)

func selectnbsend(chanType *byte, hchan chan<- any, elem any) bool
func selectnbrecv(chanType *byte, elem *any, hchan <-chan any) bool
func selectnbrecv2(chanType *byte, elem *any, received *bool, hchan <-chan any) bool

func newselect(size int) (sel *byte)
func selectsend(sel *byte, hchan chan<- any, elem *any) (selected bool)
func selectrecv(sel *byte, hchan <-chan any, elem *any) (selected bool)
func selectrecv2(sel *byte, hchan <-chan any, elem *any, received *bool) (selected bool)
func selectdefault(sel *byte) (selected bool)
func selectgo(sel *byte)
func block()

func makeslice(typ *byte, nel int64, cap int64) (ary []any)
func growslice(typ *byte, old []any, n int64) (ary []any)
func sliceslice1(old []any, lb uint64, width uint64) (ary []any)
func sliceslice(old []any, lb uint64, hb uint64, width uint64) (ary []any)
func slicearray(old *any, nel uint64, lb uint64, hb uint64, width uint64) (ary []any)

func closure() // has args, but compiler fills in

func memequal(eq *bool, size uintptr, x, y unsafe.Pointer)
func memequal8(eq *bool, size uintptr, x, y unsafe.Pointer)
func memequal16(eq *bool, size uintptr, x, y unsafe.Pointer)
func memequal32(eq *bool, size uintptr, x, y unsafe.Pointer)
func memequal64(eq *bool, size uintptr, x, y unsafe.Pointer)
func memequal128(eq *bool, size uintptr, x, y unsafe.Pointer)

// only used on 32-bit
func int64div(int64, int64) int64
func uint64div(uint64, uint64) uint64
func int64mod(int64, int64) int64
func uint64mod(uint64, uint64) uint64
func float64toint64(float64) int64
func float64touint64(float64) uint64
func int64tofloat64(int64) float64
func uint64tofloat64(uint64) float64

func complex128div(num complex128, den complex128) (quo complex128)
