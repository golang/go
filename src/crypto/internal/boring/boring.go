// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build boringcrypto && linux && (amd64 || arm64) && !android && !msan

package boring

/*
// goboringcrypto_linux_amd64.syso references pthread functions.
#cgo LDFLAGS: "-pthread"

#include "goboringcrypto.h"
*/
import "C"
import (
	"crypto/internal/boring/sig"
	_ "crypto/internal/boring/syso"
	"math/bits"
	"unsafe"
)

const available = true

func init() {
	C._goboringcrypto_BORINGSSL_bcm_power_on_self_test()
	if C._goboringcrypto_FIPS_mode() != 1 {
		panic("boringcrypto: not in FIPS mode")
	}
	sig.BoringCrypto()
}

// Unreachable marks code that should be unreachable
// when BoringCrypto is in use. It panics.
func Unreachable() {
	panic("boringcrypto: invalid code execution")
}

// provided by runtime to avoid os import.
func runtime_arg0() string

func hasSuffix(s, t string) bool {
	return len(s) > len(t) && s[len(s)-len(t):] == t
}

// UnreachableExceptTests marks code that should be unreachable
// when BoringCrypto is in use. It panics.
func UnreachableExceptTests() {
	name := runtime_arg0()
	// If BoringCrypto ran on Windows we'd need to allow _test.exe and .test.exe as well.
	if !hasSuffix(name, "_test") && !hasSuffix(name, ".test") {
		println("boringcrypto: unexpected code execution in", name)
		panic("boringcrypto: invalid code execution")
	}
}

type fail string

func (e fail) Error() string { return "boringcrypto: " + string(e) + " failed" }

func wbase(b BigInt) *C.uint8_t {
	if len(b) == 0 {
		return nil
	}
	return (*C.uint8_t)(unsafe.Pointer(&b[0]))
}

const wordBytes = bits.UintSize / 8

func bigToBN(x BigInt) *C.GO_BIGNUM {
	return C._goboringcrypto_BN_le2bn(wbase(x), C.size_t(len(x)*wordBytes), nil)
}

func bytesToBN(x []byte) *C.GO_BIGNUM {
	return C._goboringcrypto_BN_bin2bn((*C.uint8_t)(&x[0]), C.size_t(len(x)), nil)
}

func bnToBig(bn *C.GO_BIGNUM) BigInt {
	x := make(BigInt, (C._goboringcrypto_BN_num_bytes(bn)+wordBytes-1)/wordBytes)
	if C._goboringcrypto_BN_bn2le_padded(wbase(x), C.size_t(len(x)*wordBytes), bn) == 0 {
		panic("boringcrypto: bignum conversion failed")
	}
	return x
}

func bigToBn(bnp **C.GO_BIGNUM, b BigInt) bool {
	if *bnp != nil {
		C._goboringcrypto_BN_free(*bnp)
		*bnp = nil
	}
	if b == nil {
		return true
	}
	bn := bigToBN(b)
	if bn == nil {
		return false
	}
	*bnp = bn
	return true
}

// noescape hides a pointer from escape analysis.  noescape is
// the identity function but escape analysis doesn't think the
// output depends on the input.  noescape is inlined and currently
// compiles down to zero instructions.
// USE CAREFULLY!
//
//go:nosplit
func noescape(p unsafe.Pointer) unsafe.Pointer {
	x := uintptr(p)
	return unsafe.Pointer(x ^ 0)
}

var zero byte

// addr converts p to its base addr, including a noescape along the way.
// If p is nil, addr returns a non-nil pointer, so that the result can always
// be dereferenced.
//
//go:nosplit
func addr(p []byte) *byte {
	if len(p) == 0 {
		return &zero
	}
	return (*byte)(noescape(unsafe.Pointer(&p[0])))
}
