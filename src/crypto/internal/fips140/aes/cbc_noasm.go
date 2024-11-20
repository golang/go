// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (!s390x && !ppc64 && !ppc64le) || purego

package aes

func cryptBlocksEnc(b *Block, civ *[BlockSize]byte, dst, src []byte) {
	cryptBlocksEncGeneric(b, civ, dst, src)
}

func cryptBlocksDec(b *Block, civ *[BlockSize]byte, dst, src []byte) {
	cryptBlocksDecGeneric(b, civ, dst, src)
}
