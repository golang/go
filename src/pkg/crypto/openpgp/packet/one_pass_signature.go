// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packet

import (
	"crypto"
	"crypto/openpgp/error"
	"crypto/openpgp/s2k"
	"encoding/binary"
	"io"
	"os"
	"strconv"
)

// OnePassSignature represents a one-pass signature packet. See RFC 4880,
// section 5.4.
type OnePassSignature struct {
	SigType    SignatureType
	Hash       crypto.Hash
	PubKeyAlgo PublicKeyAlgorithm
	KeyId      uint64
	IsLast     bool
}

func (ops *OnePassSignature) parse(r io.Reader) (err os.Error) {
	var buf [13]byte

	_, err = readFull(r, buf[:])
	if err != nil {
		return
	}
	if buf[0] != 3 {
		err = error.UnsupportedError("one-pass-signature packet version " + strconv.Itoa(int(buf[0])))
	}

	var ok bool
	ops.Hash, ok = s2k.HashIdToHash(buf[2])
	if !ok {
		return error.UnsupportedError("hash function: " + strconv.Itoa(int(buf[2])))
	}

	ops.SigType = SignatureType(buf[1])
	ops.PubKeyAlgo = PublicKeyAlgorithm(buf[3])
	ops.KeyId = binary.BigEndian.Uint64(buf[4:12])
	ops.IsLast = buf[12] != 0
	return
}
