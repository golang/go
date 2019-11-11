// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// NOTE: If you change this file you must run "go generate"
// to update builtin.go. This is not done automatically
// to avoid depending on having a working compiler binary.

// +build ignore

package runtime

// emitted by compiler, not referred to by go programs

import "unsafe"

func newobject(typ *byte) *any
func panicdivide()
func panicshift()
func panicmakeslicelen()
func throwinit()
func panicwrap()

func gopanic(interface{})
func gorecover(*int32) interface{}
func goschedguarded()

// Note: these declarations are just for wasm port.
// Other ports call assembly stubs instead.
func goPanicIndex(x int, y int)
func goPanicIndexU(x uint, y int)
func goPanicSliceAlen(x int, y int)
func goPanicSliceAlenU(x uint, y int)
func goPanicSliceAcap(x int, y int)
func goPanicSliceAcapU(x uint, y int)
func goPanicSliceB(x int, y int)
func goPanicSliceBU(x uint, y int)
func goPanicSlice3Alen(x int, y int)
func goPanicSlice3AlenU(x uint, y int)
func goPanicSlice3Acap(x int, y int)
func goPanicSlice3AcapU(x uint, y int)
func goPanicSlice3B(x int, y int)
func goPanicSlice3BU(x uint, y int)
func goPanicSlice3C(x int, y int)
func goPanicSlice3CU(x uint, y int)

func printbool(bool)
func printfloat(float64)
func printint(int64)
func printhex(uint64)
func printuint(uint64)
func printcomplex(complex128)
func printstring(string)
func printpointer(any)
func printiface(any)
func printeface(any)
func printslice(any)
func printnl()
func printsp()
func printlock()
func printunlock()

func concatstring2(*[32]byte, string, string) string
func concatstring3(*[32]byte, string, string, string) string
func concatstring4(*[32]byte, string, string, string, string) string
func concatstring5(*[32]byte, string, string, string, string, string) string
func concatstrings(*[32]byte, []string) string

func cmpstring(string, string) int
func intstring(*[4]byte, int64) string
func slicebytetostring(*[32]byte, []byte) string
func slicebytetostringtmp([]byte) string
func slicerunetostring(*[32]byte, []rune) string
func stringtoslicebyte(*[32]byte, string) []byte
func stringtoslicerune(*[32]rune, string) []rune
func slicecopy(to any, fr any, wid uintptr) int
func slicestringcopy(to any, fr any) int

func decoderune(string, int) (retv rune, retk int)
func countrunes(string) int

// Non-empty-interface to non-empty-interface conversion.
func convI2I(typ *byte, elem any) (ret any)

// Specialized type-to-interface conversion.
// These return only a data pointer.
func convT16(val any) unsafe.Pointer     // val must be uint16-like (same size and alignment as a uint16)
func convT32(val any) unsafe.Pointer     // val must be uint32-like (same size and alignment as a uint32)
func convT64(val any) unsafe.Pointer     // val must be uint64-like (same size and alignment as a uint64 and contains no pointers)
func convTstring(val any) unsafe.Pointer // val must be a string
func convTslice(val any) unsafe.Pointer  // val must be a slice

// Type to empty-interface conversion.
func convT2E(typ *byte, elem *any) (ret any)
func convT2Enoptr(typ *byte, elem *any) (ret any)

// Type to non-empty-interface conversion.
func convT2I(tab *byte, elem *any) (ret any)
func convT2Inoptr(tab *byte, elem *any) (ret any)

// interface type assertions x.(T)
func assertE2I(typ *byte, iface any) (ret any)
func assertE2I2(typ *byte, iface any) (ret any, b bool)
func assertI2I(typ *byte, iface any) (ret any)
func assertI2I2(typ *byte, iface any) (ret any, b bool)
func panicdottypeE(have, want, iface *byte)
func panicdottypeI(have, want, iface *byte)
func panicnildottype(want *byte)

// interface equality. Type/itab pointers are already known to be equal, so
// we only need to pass one.
func ifaceeq(tab *uintptr, x, y unsafe.Pointer) (ret bool)
func efaceeq(typ *uintptr, x, y unsafe.Pointer) (ret bool)

func fastrand() uint32

// *byte is really *runtime.Type
func makemap64(mapType *byte, hint int64, mapbuf *any) (hmap map[any]any)
func makemap(mapType *byte, hint int, mapbuf *any) (hmap map[any]any)
func makemap_small() (hmap map[any]any)
func mapaccess1(mapType *byte, hmap map[any]any, key *any) (val *any)
func mapaccess1_fast32(mapType *byte, hmap map[any]any, key any) (val *any)
func mapaccess1_fast64(mapType *byte, hmap map[any]any, key any) (val *any)
func mapaccess1_faststr(mapType *byte, hmap map[any]any, key any) (val *any)
func mapaccess1_fat(mapType *byte, hmap map[any]any, key *any, zero *byte) (val *any)
func mapaccess2(mapType *byte, hmap map[any]any, key *any) (val *any, pres bool)
func mapaccess2_fast32(mapType *byte, hmap map[any]any, key any) (val *any, pres bool)
func mapaccess2_fast64(mapType *byte, hmap map[any]any, key any) (val *any, pres bool)
func mapaccess2_faststr(mapType *byte, hmap map[any]any, key any) (val *any, pres bool)
func mapaccess2_fat(mapType *byte, hmap map[any]any, key *any, zero *byte) (val *any, pres bool)
func mapassign(mapType *byte, hmap map[any]any, key *any) (val *any)
func mapassign_fast32(mapType *byte, hmap map[any]any, key any) (val *any)
func mapassign_fast32ptr(mapType *byte, hmap map[any]any, key any) (val *any)
func mapassign_fast64(mapType *byte, hmap map[any]any, key any) (val *any)
func mapassign_fast64ptr(mapType *byte, hmap map[any]any, key any) (val *any)
func mapassign_faststr(mapType *byte, hmap map[any]any, key any) (val *any)
func mapiterinit(mapType *byte, hmap map[any]any, hiter *any)
func mapdelete(mapType *byte, hmap map[any]any, key *any)
func mapdelete_fast32(mapType *byte, hmap map[any]any, key any)
func mapdelete_fast64(mapType *byte, hmap map[any]any, key any)
func mapdelete_faststr(mapType *byte, hmap map[any]any, key any)
func mapiternext(hiter *any)
func mapclear(mapType *byte, hmap map[any]any)

// *byte is really *runtime.Type
func makechan64(chanType *byte, size int64) (hchan chan any)
func makechan(chanType *byte, size int) (hchan chan any)
func chanrecv1(hchan <-chan any, elem *any)
func chanrecv2(hchan <-chan any, elem *any) bool
func chansend1(hchan chan<- any, elem *any)
func closechan(hchan any)

var writeBarrier struct {
	enabled bool
	pad     [3]byte
	needed  bool
	cgo     bool
	alignme uint64
}

// *byte is really *runtime.Type
func typedmemmove(typ *byte, dst *any, src *any)
func typedmemclr(typ *byte, dst *any)
func typedslicecopy(typ *byte, dst any, src any) int

func selectnbsend(hchan chan<- any, elem *any) bool
func selectnbrecv(elem *any, hchan <-chan any) bool
func selectnbrecv2(elem *any, received *bool, hchan <-chan any) bool

func selectsetpc(cas *byte)
func selectgo(cas0 *byte, order0 *byte, ncases int) (int, bool)
func block()

func makeslice(typ *byte, len int, cap int) unsafe.Pointer
func makeslice64(typ *byte, len int64, cap int64) unsafe.Pointer
func growslice(typ *byte, old []any, cap int) (ary []any)
func memmove(to *any, frm *any, length uintptr)
func memclrNoHeapPointers(ptr unsafe.Pointer, n uintptr)
func memclrHasPointers(ptr unsafe.Pointer, n uintptr)

func memequal(x, y *any, size uintptr) bool
func memequal0(x, y *any) bool
func memequal8(x, y *any) bool
func memequal16(x, y *any) bool
func memequal32(x, y *any) bool
func memequal64(x, y *any) bool
func memequal128(x, y *any) bool
func f32equal(p, q unsafe.Pointer) bool
func f64equal(p, q unsafe.Pointer) bool
func c64equal(p, q unsafe.Pointer) bool
func c128equal(p, q unsafe.Pointer) bool
func strequal(p, q unsafe.Pointer) bool
func interequal(p, q unsafe.Pointer) bool
func nilinterequal(p, q unsafe.Pointer) bool

func memhash(p unsafe.Pointer, h uintptr, size uintptr) uintptr
func memhash0(p unsafe.Pointer, h uintptr) uintptr
func memhash8(p unsafe.Pointer, h uintptr) uintptr
func memhash16(p unsafe.Pointer, h uintptr) uintptr
func memhash32(p unsafe.Pointer, h uintptr) uintptr
func memhash64(p unsafe.Pointer, h uintptr) uintptr
func memhash128(p unsafe.Pointer, h uintptr) uintptr
func f32hash(p unsafe.Pointer, h uintptr) uintptr
func f64hash(p unsafe.Pointer, h uintptr) uintptr
func c64hash(p unsafe.Pointer, h uintptr) uintptr
func c128hash(p unsafe.Pointer, h uintptr) uintptr
func strhash(a unsafe.Pointer, h uintptr) uintptr
func interhash(p unsafe.Pointer, h uintptr) uintptr
func nilinterhash(p unsafe.Pointer, h uintptr) uintptr

// only used on 32-bit
func int64div(int64, int64) int64
func uint64div(uint64, uint64) uint64
func int64mod(int64, int64) int64
func uint64mod(uint64, uint64) uint64
func float64toint64(float64) int64
func float64touint64(float64) uint64
func float64touint32(float64) uint32
func int64tofloat64(int64) float64
func uint64tofloat64(uint64) float64
func uint32tofloat64(uint32) float64

func complex128div(num complex128, den complex128) (quo complex128)

// race detection
func racefuncenter(uintptr)
func racefuncenterfp()
func racefuncexit()
func raceread(uintptr)
func racewrite(uintptr)
func racereadrange(addr, size uintptr)
func racewriterange(addr, size uintptr)

// memory sanitizer
func msanread(addr, size uintptr)
func msanwrite(addr, size uintptr)

func checkptrAlignment(unsafe.Pointer, *byte, uintptr)
func checkptrArithmetic(unsafe.Pointer, []unsafe.Pointer)

func libfuzzerTraceCmp1(uint8, uint8)
func libfuzzerTraceCmp2(uint16, uint16)
func libfuzzerTraceCmp4(uint32, uint32)
func libfuzzerTraceCmp8(uint64, uint64)
func libfuzzerTraceConstCmp1(uint8, uint8)
func libfuzzerTraceConstCmp2(uint16, uint16)
func libfuzzerTraceConstCmp4(uint32, uint32)
func libfuzzerTraceConstCmp8(uint64, uint64)

// architecture variants
var x86HasPOPCNT bool
var x86HasSSE41 bool
var x86HasFMA bool
var armHasVFPv4 bool
var arm64HasATOMICS bool
