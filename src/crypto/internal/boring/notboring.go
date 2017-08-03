// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !linux !amd64 cmd_go_bootstrap

package boring

const available = false

// Unreachable marks code that should be unreachable
// when BoringCrypto is in use. It is a no-op without BoringCrypto.
func Unreachable() {}

// UnreachableExceptTests marks code that should be unreachable
// when BoringCrypto is in use. It is a no-op without BoringCrypto.
func UnreachableExceptTests() {}

type randReader int

func (randReader) Read(b []byte) (int, error) {
	panic("boringcrypto: not available")
}

const RandReader = randReader(0)
