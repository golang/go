// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// NOTE: If you change this file you must run "go generate"
// in cmd/compile/internal/typecheck
// to update builtin.go. This is not done automatically
// to avoid depending on having a working compiler binary.

//go:build ignore

package runtime

// emitted by compiler, not referred to by go programs

import "unsafe"

func newobject(typ *byte) *any
func mallocgc(size uintptr, typ *byte, needszero bool) unsafe.Pointer
func panicdivide()
func panicshift()
func panicmakeslicelen()
func panicmakeslicecap()
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
func goPanicSliceConvert(x int, y int)

func printbool(bool)
func printfloat(float64)
func printint(int64)
func printhex(uint64)
func printuint(uint64)
func printcomplex(complex128)
func printstring(string)
func printpointer(any)
func printuintptr(uintptr)
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

func concatbyte2(string, string) []byte
func concatbyte3(string, string, string) []byte
func concatbyte4(string, string, string, string) []byte
func concatbyte5(string, string, string, string, string) []byte
func concatbytes([]string) []byte

func cmpstring(string, string) int
func intstring(*[4]byte, int64) string
func slicebytetostring(buf *[32]byte, ptr *byte, n int) string
func slicebytetostringtmp(ptr *byte, n int) string
func slicerunetostring(*[32]byte, []rune) string
func stringtoslicebyte(*[32]byte, string) []byte
func stringtoslicerune(*[32]rune, string) []rune
func slicecopy(toPtr *any, toLen int, fromPtr *any, fromLen int, wid uintptr) int

func decoderune(string, int) (retv rune, retk int)
func countrunes(string) int

// Convert non-interface type to the data word of a (empty or nonempty) interface.
func convT(typ *byte, elem *any) unsafe.Pointer

// Same as convT, for types with no pointers in them.
func convTnoptr(typ *byte, elem *any) unsafe.Pointer

// Specialized versions of convT for specific types.
// These functions take concrete types in the runtime. But they may
// be used for a wider range of types, which have the same memory
// layout as the parameter type. The compiler converts the
// to-be-converted type to the parameter type before calling the
// runtime function. This way, the call is ABI-insensitive.
func convT16(val uint16) unsafe.Pointer
func convT32(val uint32) unsafe.Pointer
func convT64(val uint64) unsafe.Pointer
func convTstring(val string) unsafe.Pointer
func convTslice(val []uint8) unsafe.Pointer

// interface type assertions x.(T)
func assertE2I(inter *byte, typ *byte) *byte
func assertE2I2(inter *byte, typ *byte) *byte
func panicdottypeE(have, want, iface *byte)
func panicdottypeI(have, want, iface *byte)
func panicnildottype(want *byte)
func typeAssert(s *byte, typ *byte) *byte

// interface switches
func interfaceSwitch(s *byte, t *byte) (int, *byte)

// interface equality. Type/itab pointers are already known to be equal, so
// we only need to pass one.
func ifaceeq(tab *uintptr, x, y unsafe.Pointer) (ret bool)
func efaceeq(typ *uintptr, x, y unsafe.Pointer) (ret bool)

// panic for various rangefunc iterator errors
func panicrangestate(state int)

// defer in range over func
func deferrangefunc() interface{}

func rand() uint64
func rand32() uint32

// *byte is really *runtime.Type
func makemap64(mapType *byte, hint int64, mapbuf *any) (hmap map[any]any)
func makemap(mapType *byte, hint int, mapbuf *any) (hmap map[any]any)
func makemap_small() (hmap map[any]any)
func mapaccess1(mapType *byte, hmap map[any]any, key *any) (val *any)
func mapaccess1_fast32(mapType *byte, hmap map[any]any, key uint32) (val *any)
func mapaccess1_fast64(mapType *byte, hmap map[any]any, key uint64) (val *any)
func mapaccess1_faststr(mapType *byte, hmap map[any]any, key string) (val *any)
func mapaccess1_fat(mapType *byte, hmap map[any]any, key *any, zero *byte) (val *any)
func mapaccess2(mapType *byte, hmap map[any]any, key *any) (val *any, pres bool)
func mapaccess2_fast32(mapType *byte, hmap map[any]any, key uint32) (val *any, pres bool)
func mapaccess2_fast64(mapType *byte, hmap map[any]any, key uint64) (val *any, pres bool)
func mapaccess2_faststr(mapType *byte, hmap map[any]any, key string) (val *any, pres bool)
func mapaccess2_fat(mapType *byte, hmap map[any]any, key *any, zero *byte) (val *any, pres bool)
func mapassign(mapType *byte, hmap map[any]any, key *any) (val *any)
func mapassign_fast32(mapType *byte, hmap map[any]any, key uint32) (val *any)
func mapassign_fast32ptr(mapType *byte, hmap map[any]any, key unsafe.Pointer) (val *any)
func mapassign_fast64(mapType *byte, hmap map[any]any, key uint64) (val *any)
func mapassign_fast64ptr(mapType *byte, hmap map[any]any, key unsafe.Pointer) (val *any)
func mapassign_faststr(mapType *byte, hmap map[any]any, key string) (val *any)
func mapiterinit(mapType *byte, hmap map[any]any, hiter *any)
func mapdelete(mapType *byte, hmap map[any]any, key *any)
func mapdelete_fast32(mapType *byte, hmap map[any]any, key uint32)
func mapdelete_fast64(mapType *byte, hmap map[any]any, key uint64)
func mapdelete_faststr(mapType *byte, hmap map[any]any, key string)
func mapiternext(hiter *any)
func mapclear(mapType *byte, hmap map[any]any)

// *byte is really *runtime.Type
func makechan64(chanType *byte, size int64) (hchan chan any)
func makechan(chanType *byte, size int) (hchan chan any)
func chanrecv1(hchan <-chan any, elem *any)
func chanrecv2(hchan <-chan any, elem *any) bool
func chansend1(hchan chan<- any, elem *any)
func closechan(hchan chan<- any)
func chanlen(hchan any) int
func chancap(hchan any) int

var writeBarrier struct {
	enabled bool
	pad     [3]byte
	cgo     bool
	alignme uint64
}

// *byte is really *runtime.Type
func typedmemmove(typ *byte, dst *any, src *any)
func typedmemclr(typ *byte, dst *any)
func typedslicecopy(typ *byte, dstPtr *any, dstLen int, srcPtr *any, srcLen int) int

func selectnbsend(hchan chan<- any, elem *any) bool
func selectnbrecv(elem *any, hchan <-chan any) (bool, bool)

func selectsetpc(pc *uintptr)
func selectgo(cas0 *byte, order0 *byte, pc0 *uintptr, nsends int, nrecvs int, block bool) (int, bool)
func block()

func makeslice(typ *byte, len int, cap int) unsafe.Pointer
func makeslice64(typ *byte, len int64, cap int64) unsafe.Pointer
func makeslicecopy(typ *byte, tolen int, fromlen int, from unsafe.Pointer) unsafe.Pointer
func growslice(oldPtr *any, newLen, oldCap, num int, et *byte) (ary []any)
func unsafeslicecheckptr(typ *byte, ptr unsafe.Pointer, len int64)
func panicunsafeslicelen()
func panicunsafeslicenilptr()
func unsafestringcheckptr(ptr unsafe.Pointer, len int64)
func panicunsafestringlen()
func panicunsafestringnilptr()

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

func memhash(x *any, h uintptr, size uintptr) uintptr
func memhash0(p unsafe.Pointer, h uintptr) uintptr
func memhash8(p unsafe.Pointer, h uintptr) uintptr
func memhash16(p unsafe.Pointer, h uintptr) uintptr
func memhash32(p unsafe.Pointer, h uintptr) uintptr
func memhash64(p unsafe.Pointer, h uintptr) uintptr
func memhash128(p unsafe.Pointer, h uintptr) uintptr
func f32hash(p *any, h uintptr) uintptr
func f64hash(p *any, h uintptr) uintptr
func c64hash(p *any, h uintptr) uintptr
func c128hash(p *any, h uintptr) uintptr
func strhash(a *any, h uintptr) uintptr
func interhash(p *any, h uintptr) uintptr
func nilinterhash(p *any, h uintptr) uintptr

// only used on 32-bit
func int64div(int64, int64) int64
func uint64div(uint64, uint64) uint64
func int64mod(int64, int64) int64
func uint64mod(uint64, uint64) uint64
func float64toint64(float64) int64
func float64touint64(float64) uint64
func float64touint32(float64) uint32
func int64tofloat64(int64) float64
func int64tofloat32(int64) float32
func uint64tofloat64(uint64) float64
func uint64tofloat32(uint64) float32
func uint32tofloat64(uint32) float64

func complex128div(num complex128, den complex128) (quo complex128)

// race detection
func racefuncenter(uintptr)
func racefuncexit()
func raceread(uintptr)
func racewrite(uintptr)
func racereadrange(addr, size uintptr)
func racewriterange(addr, size uintptr)

// memory sanitizer
func msanread(addr, size uintptr)
func msanwrite(addr, size uintptr)
func msanmove(dst, src, size uintptr)

// address sanitizer
func asanread(addr, size uintptr)
func asanwrite(addr, size uintptr)

func checkptrAlignment(unsafe.Pointer, *byte, uintptr)
func checkptrArithmetic(unsafe.Pointer, []unsafe.Pointer)

func libfuzzerTraceCmp1(uint8, uint8, uint)
func libfuzzerTraceCmp2(uint16, uint16, uint)
func libfuzzerTraceCmp4(uint32, uint32, uint)
func libfuzzerTraceCmp8(uint64, uint64, uint)
func libfuzzerTraceConstCmp1(uint8, uint8, uint)
func libfuzzerTraceConstCmp2(uint16, uint16, uint)
func libfuzzerTraceConstCmp4(uint32, uint32, uint)
func libfuzzerTraceConstCmp8(uint64, uint64, uint)
func libfuzzerHookStrCmp(string, string, uint)
func libfuzzerHookEqualFold(string, string, uint)

func addCovMeta(p unsafe.Pointer, len uint32, hash [16]byte, pkpath string, pkgId int, cmode uint8, cgran uint8) uint32

// architecture variants
var x86HasPOPCNT bool
var x86HasSSE41 bool
var x86HasFMA bool
var armHasVFPv4 bool
var arm64HasATOMICS bool

func asanregisterglobals(unsafe.Pointer, uintptr)
