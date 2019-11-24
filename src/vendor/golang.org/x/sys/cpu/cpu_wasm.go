// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build wasm

package cpu

// We're compiling the cpu package for an unknown (software-abstracted) CPU.
// Make CacheLinePad an empty struct and hope that the usual struct alignment
// rules are good enough.

const cacheLineSize = 0

func doinit() {}
