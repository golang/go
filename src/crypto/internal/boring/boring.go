// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux,amd64
// +build !android
// +build !cmd_go_bootstrap
// +build !msan

package boring

import (
	"crypto/internal/boring/boringcrypto"
	"crypto/internal/boring/sig"
)

type externalCrypto interface {
	Init()

	aes
	ecdsa
	hmac
	rsa
	sha
}

var external externalCrypto

const available = true

func init() {
	external = boringcrypto.NewBoringCrypto()
	external.Init()
	sig.BoringCrypto()
}

type fail string

func (e fail) Error() string { return "boringcrypto: " + string(e) + " failed" }

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
