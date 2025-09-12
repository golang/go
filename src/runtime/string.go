// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/abi"
	"internal/bytealg"
	"internal/goarch"
	"internal/runtime/math"
	"internal/runtime/strconv"
	"internal/runtime/sys"
	"unsafe"
)

// The constant is known to the compiler.
// There is no fundamental theory behind this number.
const tmpStringBufSize = 32

type tmpBuf [tmpStringBufSize]byte

// concatstrings implements a Go string concatenation x+y+z+...
// The operands are passed in the slice a.
// If buf != nil, the compiler has determined that the result does not
// escape the calling function, so the string data can be stored in buf
// if small enough.
func concatstrings(buf *tmpBuf, a []string) string {
	idx := 0
	l := 0
	count := 0
	for i, x := range a {
		n := len(x)
		if n == 0 {
			continue
		}
		if l+n < l {
			throw("string concatenation too long")
		}
		l += n
		count++
		idx = i
	}
	if count == 0 {
		return ""
	}

	// If there is just one string and either it is not on the stack
	// or our result does not escape the calling frame (buf != nil),
	// then we can return that string directly.
	if count == 1 && (buf != nil || !stringDataOnStack(a[idx])) {
		return a[idx]
	}
	s, b := rawstringtmp(buf, l)
	for _, x := range a {
		n := copy(b, x)
		b = b[n:]
	}
	return s
}

// concatstring2 helps make the callsite smaller (compared to concatstrings),
// and we think this is currently more valuable than omitting one call in the
// chain, the same goes for concatstring{3,4,5}.
func concatstring2(buf *tmpBuf, a0, a1 string) string {
	return concatstrings(buf, []string{a0, a1})
}

func concatstring3(buf *tmpBuf, a0, a1, a2 string) string {
	return concatstrings(buf, []string{a0, a1, a2})
}

func concatstring4(buf *tmpBuf, a0, a1, a2, a3 string) string {
	return concatstrings(buf, []string{a0, a1, a2, a3})
}

func concatstring5(buf *tmpBuf, a0, a1, a2, a3, a4 string) string {
	return concatstrings(buf, []string{a0, a1, a2, a3, a4})
}

// concatbytes implements a Go string concatenation x+y+z+... returning a slice
// of bytes.
// The operands are passed in the slice a.
func concatbytes(buf *tmpBuf, a []string) []byte {
	l := 0
	for _, x := range a {
		n := len(x)
		if l+n < l {
			throw("string concatenation too long")
		}
		l += n
	}
	if l == 0 {
		// This is to match the return type of the non-optimized concatenation.
		return []byte{}
	}

	var b []byte
	if buf != nil && l <= len(buf) {
		*buf = tmpBuf{}
		b = buf[:l]
	} else {
		b = rawbyteslice(l)
	}
	offset := 0
	for _, x := range a {
		copy(b[offset:], x)
		offset += len(x)
	}

	return b
}

// concatbyte2 helps make the callsite smaller (compared to concatbytes),
// and we think this is currently more valuable than omitting one call in
// the chain, the same goes for concatbyte{3,4,5}.
func concatbyte2(buf *tmpBuf, a0, a1 string) []byte {
	return concatbytes(buf, []string{a0, a1})
}

func concatbyte3(buf *tmpBuf, a0, a1, a2 string) []byte {
	return concatbytes(buf, []string{a0, a1, a2})
}

func concatbyte4(buf *tmpBuf, a0, a1, a2, a3 string) []byte {
	return concatbytes(buf, []string{a0, a1, a2, a3})
}

func concatbyte5(buf *tmpBuf, a0, a1, a2, a3, a4 string) []byte {
	return concatbytes(buf, []string{a0, a1, a2, a3, a4})
}

// slicebytetostring converts a byte slice to a string.
// It is inserted by the compiler into generated code.
// ptr is a pointer to the first element of the slice;
// n is the length of the slice.
// Buf is a fixed-size buffer for the result,
// it is not nil if the result does not escape.
func slicebytetostring(buf *tmpBuf, ptr *byte, n int) string {
	if n == 0 {
		// Turns out to be a relatively common case.
		// Consider that you want to parse out data between parens in "foo()bar",
		// you find the indices and convert the subslice to string.
		return ""
	}
	if raceenabled {
		racereadrangepc(unsafe.Pointer(ptr),
			uintptr(n),
			sys.GetCallerPC(),
			abi.FuncPCABIInternal(slicebytetostring))
	}
	if msanenabled {
		msanread(unsafe.Pointer(ptr), uintptr(n))
	}
	if asanenabled {
		asanread(unsafe.Pointer(ptr), uintptr(n))
	}
	if n == 1 {
		p := unsafe.Pointer(&staticuint64s[*ptr])
		if goarch.BigEndian {
			p = add(p, 7)
		}
		return unsafe.String((*byte)(p), 1)
	}

	var p unsafe.Pointer
	if buf != nil && n <= len(buf) {
		p = unsafe.Pointer(buf)
	} else {
		p = mallocgc(uintptr(n), nil, false)
	}
	memmove(p, unsafe.Pointer(ptr), uintptr(n))
	return unsafe.String((*byte)(p), n)
}

// stringDataOnStack reports whether the string's data is
// stored on the current goroutine's stack.
func stringDataOnStack(s string) bool {
	ptr := uintptr(unsafe.Pointer(unsafe.StringData(s)))
	stk := getg().stack
	return stk.lo <= ptr && ptr < stk.hi
}

func rawstringtmp(buf *tmpBuf, l int) (s string, b []byte) {
	if buf != nil && l <= len(buf) {
		b = buf[:l]
		s = slicebytetostringtmp(&b[0], len(b))
	} else {
		s, b = rawstring(l)
	}
	return
}

// slicebytetostringtmp returns a "string" referring to the actual []byte bytes.
//
// Callers need to ensure that the returned string will not be used after
// the calling goroutine modifies the original slice or synchronizes with
// another goroutine.
//
// The function is only called when instrumenting
// and otherwise intrinsified by the compiler.
//
// Some internal compiler optimizations use this function.
//   - Used for m[T1{... Tn{..., string(k), ...} ...}] and m[string(k)]
//     where k is []byte, T1 to Tn is a nesting of struct and array literals.
//   - Used for "<"+string(b)+">" concatenation where b is []byte.
//   - Used for string(b)=="foo" comparison where b is []byte.
func slicebytetostringtmp(ptr *byte, n int) string {
	if raceenabled && n > 0 {
		racereadrangepc(unsafe.Pointer(ptr),
			uintptr(n),
			sys.GetCallerPC(),
			abi.FuncPCABIInternal(slicebytetostringtmp))
	}
	if msanenabled && n > 0 {
		msanread(unsafe.Pointer(ptr), uintptr(n))
	}
	if asanenabled && n > 0 {
		asanread(unsafe.Pointer(ptr), uintptr(n))
	}
	return unsafe.String(ptr, n)
}

func stringtoslicebyte(buf *tmpBuf, s string) []byte {
	var b []byte
	if buf != nil && len(s) <= len(buf) {
		*buf = tmpBuf{}
		b = buf[:len(s)]
	} else {
		b = rawbyteslice(len(s))
	}
	copy(b, s)
	return b
}

func stringtoslicerune(buf *[tmpStringBufSize]rune, s string) []rune {
	// two passes.
	// unlike slicerunetostring, no race because strings are immutable.
	n := 0
	for range s {
		n++
	}

	var a []rune
	if buf != nil && n <= len(buf) {
		*buf = [tmpStringBufSize]rune{}
		a = buf[:n]
	} else {
		a = rawruneslice(n)
	}

	n = 0
	for _, r := range s {
		a[n] = r
		n++
	}
	return a
}

func slicerunetostring(buf *tmpBuf, a []rune) string {
	if raceenabled && len(a) > 0 {
		racereadrangepc(unsafe.Pointer(&a[0]),
			uintptr(len(a))*unsafe.Sizeof(a[0]),
			sys.GetCallerPC(),
			abi.FuncPCABIInternal(slicerunetostring))
	}
	if msanenabled && len(a) > 0 {
		msanread(unsafe.Pointer(&a[0]), uintptr(len(a))*unsafe.Sizeof(a[0]))
	}
	if asanenabled && len(a) > 0 {
		asanread(unsafe.Pointer(&a[0]), uintptr(len(a))*unsafe.Sizeof(a[0]))
	}
	var dum [4]byte
	size1 := 0
	for _, r := range a {
		size1 += encoderune(dum[:], r)
	}
	s, b := rawstringtmp(buf, size1+3)
	size2 := 0
	for _, r := range a {
		// check for race
		if size2 >= size1 {
			break
		}
		size2 += encoderune(b[size2:], r)
	}
	return s[:size2]
}

type stringStruct struct {
	str unsafe.Pointer
	len int
}

// Variant with *byte pointer type for DWARF debugging.
type stringStructDWARF struct {
	str *byte
	len int
}

func stringStructOf(sp *string) *stringStruct {
	return (*stringStruct)(unsafe.Pointer(sp))
}

func intstring(buf *[4]byte, v int64) (s string) {
	var b []byte
	if buf != nil {
		b = buf[:]
		s = slicebytetostringtmp(&b[0], len(b))
	} else {
		s, b = rawstring(4)
	}
	if int64(rune(v)) != v {
		v = runeError
	}
	n := encoderune(b, rune(v))
	return s[:n]
}

// rawstring allocates storage for a new string. The returned
// string and byte slice both refer to the same storage.
// The storage is not zeroed. Callers should use
// b to set the string contents and then drop b.
func rawstring(size int) (s string, b []byte) {
	p := mallocgc(uintptr(size), nil, false)
	return unsafe.String((*byte)(p), size), unsafe.Slice((*byte)(p), size)
}

// rawbyteslice allocates a new byte slice. The byte slice is not zeroed.
func rawbyteslice(size int) (b []byte) {
	cap := roundupsize(uintptr(size), true)
	p := mallocgc(cap, nil, false)
	if cap != uintptr(size) {
		memclrNoHeapPointers(add(p, uintptr(size)), cap-uintptr(size))
	}

	*(*slice)(unsafe.Pointer(&b)) = slice{p, size, int(cap)}
	return
}

// rawruneslice allocates a new rune slice. The rune slice is not zeroed.
func rawruneslice(size int) (b []rune) {
	if uintptr(size) > maxAlloc/4 {
		throw("out of memory")
	}
	mem := roundupsize(uintptr(size)*4, true)
	p := mallocgc(mem, nil, false)
	if mem != uintptr(size)*4 {
		memclrNoHeapPointers(add(p, uintptr(size)*4), mem-uintptr(size)*4)
	}

	*(*slice)(unsafe.Pointer(&b)) = slice{p, size, int(mem / 4)}
	return
}

// used by cmd/cgo
func gobytes(p *byte, n int) (b []byte) {
	if n == 0 {
		return make([]byte, 0)
	}

	if n < 0 || uintptr(n) > maxAlloc {
		panic(errorString("gobytes: length out of range"))
	}

	bp := mallocgc(uintptr(n), nil, false)
	memmove(bp, unsafe.Pointer(p), uintptr(n))

	*(*slice)(unsafe.Pointer(&b)) = slice{bp, n, n}
	return
}

// This is exported via linkname to assembly in syscall (for Plan9) and cgo.
//
//go:linkname gostring
func gostring(p *byte) string {
	l := findnull(p)
	if l == 0 {
		return ""
	}
	s, b := rawstring(l)
	memmove(unsafe.Pointer(&b[0]), unsafe.Pointer(p), uintptr(l))
	return s
}

// internal_syscall_gostring is a version of gostring for internal/syscall/unix.
//
//go:linkname internal_syscall_gostring internal/syscall/unix.gostring
func internal_syscall_gostring(p *byte) string {
	return gostring(p)
}

func gostringn(p *byte, l int) string {
	if l == 0 {
		return ""
	}
	s, b := rawstring(l)
	memmove(unsafe.Pointer(&b[0]), unsafe.Pointer(p), uintptr(l))
	return s
}

// parseByteCount parses a string that represents a count of bytes.
//
// s must match the following regular expression:
//
//	^[0-9]+(([KMGT]i)?B)?$
//
// In other words, an integer byte count with an optional unit
// suffix. Acceptable suffixes include one of
// - KiB, MiB, GiB, TiB which represent binary IEC/ISO 80000 units, or
// - B, which just represents bytes.
//
// Returns an int64 because that's what its callers want and receive,
// but the result is always non-negative.
func parseByteCount(s string) (int64, bool) {
	// The empty string is not valid.
	if s == "" {
		return 0, false
	}
	// Handle the easy non-suffix case.
	last := s[len(s)-1]
	if last >= '0' && last <= '9' {
		n, ok := strconv.Atoi64(s)
		if !ok || n < 0 {
			return 0, false
		}
		return n, ok
	}
	// Failing a trailing digit, this must always end in 'B'.
	// Also at this point there must be at least one digit before
	// that B.
	if last != 'B' || len(s) < 2 {
		return 0, false
	}
	// The one before that must always be a digit or 'i'.
	if c := s[len(s)-2]; c >= '0' && c <= '9' {
		// Trivial 'B' suffix.
		n, ok := strconv.Atoi64(s[:len(s)-1])
		if !ok || n < 0 {
			return 0, false
		}
		return n, ok
	} else if c != 'i' {
		return 0, false
	}
	// Finally, we need at least 4 characters now, for the unit
	// prefix and at least one digit.
	if len(s) < 4 {
		return 0, false
	}
	power := 0
	switch s[len(s)-3] {
	case 'K':
		power = 1
	case 'M':
		power = 2
	case 'G':
		power = 3
	case 'T':
		power = 4
	default:
		// Invalid suffix.
		return 0, false
	}
	m := uint64(1)
	for i := 0; i < power; i++ {
		m *= 1024
	}
	n, ok := strconv.Atoi64(s[:len(s)-3])
	if !ok || n < 0 {
		return 0, false
	}
	un := uint64(n)
	if un > math.MaxUint64/m {
		// Overflow.
		return 0, false
	}
	un *= m
	if un > uint64(math.MaxInt64) {
		// Overflow.
		return 0, false
	}
	return int64(un), true
}

//go:nosplit
func findnull(s *byte) int {
	if s == nil {
		return 0
	}

	// Avoid IndexByteString on Plan 9 because it uses SSE instructions
	// on x86 machines, and those are classified as floating point instructions,
	// which are illegal in a note handler.
	if GOOS == "plan9" {
		p := (*[maxAlloc/2 - 1]byte)(unsafe.Pointer(s))
		l := 0
		for p[l] != 0 {
			l++
		}
		return l
	}

	// pageSize is the unit we scan at a time looking for NULL.
	// It must be the minimum page size for any architecture Go
	// runs on. It's okay (just a minor performance loss) if the
	// actual system page size is larger than this value.
	const pageSize = 4096

	offset := 0
	ptr := unsafe.Pointer(s)
	// IndexByteString uses wide reads, so we need to be careful
	// with page boundaries. Call IndexByteString on
	// [ptr, endOfPage) interval.
	safeLen := int(pageSize - uintptr(ptr)%pageSize)

	for {
		t := *(*string)(unsafe.Pointer(&stringStruct{ptr, safeLen}))
		// Check one page at a time.
		if i := bytealg.IndexByteString(t, 0); i != -1 {
			return offset + i
		}
		// Move to next page
		ptr = unsafe.Pointer(uintptr(ptr) + uintptr(safeLen))
		offset += safeLen
		safeLen = pageSize
	}
}

func findnullw(s *uint16) int {
	if s == nil {
		return 0
	}
	p := (*[maxAlloc/2/2 - 1]uint16)(unsafe.Pointer(s))
	l := 0
	for p[l] != 0 {
		l++
	}
	return l
}

//go:nosplit
func gostringnocopy(str *byte) string {
	ss := stringStruct{str: unsafe.Pointer(str), len: findnull(str)}
	s := *(*string)(unsafe.Pointer(&ss))
	return s
}

func gostringw(strw *uint16) string {
	var buf [8]byte
	str := (*[maxAlloc/2/2 - 1]uint16)(unsafe.Pointer(strw))
	n1 := 0
	for i := 0; str[i] != 0; i++ {
		n1 += encoderune(buf[:], rune(str[i]))
	}
	s, b := rawstring(n1 + 4)
	n2 := 0
	for i := 0; str[i] != 0; i++ {
		// check for race
		if n2 >= n1 {
			break
		}
		n2 += encoderune(b[n2:], rune(str[i]))
	}
	b[n2] = 0 // for luck
	return s[:n2]
}
