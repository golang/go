// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (!amd64 && !386 && !ppc64le) || purego

package sha3

// A storageBuf is an aligned array of maxRate bytes.
type storageBuf [maxRate]byte

func (b *storageBuf) asBytes() *[maxRate]byte {
	return (*[maxRate]byte)(b)
}

var (
	xorIn            = xorInGeneric
	copyOut          = copyOutGeneric
	xorInUnaligned   = xorInGeneric
	copyOutUnaligned = copyOutGeneric
)

const xorImplementationUnaligned = "generic"
