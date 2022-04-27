// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build boringcrypto && linux && amd64 && !android && !cmd_go_bootstrap && !msan
// +build boringcrypto,linux,amd64,!android,!cmd_go_bootstrap,!msan

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
	"math/big"
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

// provided by runtime to avoid os import
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

func wbase(b []big.Word) *C.uint8_t {
	if len(b) == 0 {
		return nil
	}
	return (*C.uint8_t)(unsafe.Pointer(&b[0]))
}

const wordBytes = bits.UintSize / 8

func bigToBN(x *big.Int) *C.GO_BIGNUM {
	raw := x.Bits()
	return C._goboringcrypto_BN_le2bn(wbase(raw), C.size_t(len(raw)*wordBytes), nil)
}

func bnToBig(bn *C.GO_BIGNUM) *big.Int {
	raw := make([]big.Word, (C._goboringcrypto_BN_num_bytes(bn)+wordBytes-1)/wordBytes)
	if C._goboringcrypto_BN_bn2le_padded(wbase(raw), C.size_t(len(raw)*wordBytes), bn) == 0 {
		panic("boringcrypto: bignum conversion failed")
	}
	return new(big.Int).SetBits(raw)
}

func bigToBn(bnp **C.GO_BIGNUM, b *big.Int) bool {
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
