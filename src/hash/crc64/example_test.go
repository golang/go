// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package crc64_test

import (
	"fmt"
	"hash/crc64"
)

func ExampleChecksum() {
	table := crc64.MakeTable(crc64.ECMA)
	checksum := crc64.Checksum([]byte("Hello, world!"), table)
	fmt.Printf("%016x\n", checksum)
	// Output:
	// 8e59e143665877c4
}
