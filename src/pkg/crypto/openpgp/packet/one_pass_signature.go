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

const onePassSignatureVersion = 3

func (ops *OnePassSignature) parse(r io.Reader) (err os.Error) {
	var buf [13]byte

	_, err = readFull(r, buf[:])
	if err != nil {
		return
	}
	if buf[0] != onePassSignatureVersion {
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

// Serialize marshals the given OnePassSignature to w.
func (ops *OnePassSignature) Serialize(w io.Writer) os.Error {
	var buf [13]byte
	buf[0] = onePassSignatureVersion
	buf[1] = uint8(ops.SigType)
	var ok bool
	buf[2], ok = s2k.HashToHashId(ops.Hash)
	if !ok {
		return error.UnsupportedError("hash type: " + strconv.Itoa(int(ops.Hash)))
	}
	buf[3] = uint8(ops.PubKeyAlgo)
	binary.BigEndian.PutUint64(buf[4:12], ops.KeyId)
	if ops.IsLast {
		buf[12] = 1
	}

	if err := serializeHeader(w, packetTypeOnePassSignature, len(buf)); err != nil {
		return err
	}
	_, err := w.Write(buf[:])
	return err
}
