// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sha512

import (
	"internal/cpu"
	"unsafe"
)

var useAsm = cpu.S390X.HasSHA512

func doBlockGeneric(dig *digest, p *byte, n int) {
	blockGeneric(dig, unsafe.String(p, n))
}
