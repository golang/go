// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package abi

// ZeroValSize is the size in bytes of [ZeroVal].
const ZeroValSize = 1024

// ZeroVal is a region containing all zero bytes.
var ZeroVal [ZeroValSize]byte
