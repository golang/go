// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packet

import (
	"crypto/openpgp/error"
	"crypto/rand"
	"crypto/rsa"
	"encoding/binary"
	"io"
	"os"
	"strconv"
)

// EncryptedKey represents a public-key encrypted session key. See RFC 4880,
// section 5.1.
type EncryptedKey struct {
	KeyId      uint64
	Algo       PublicKeyAlgorithm
	Encrypted  []byte
	CipherFunc CipherFunction // only valid after a successful Decrypt
	Key        []byte         // only valid after a successful Decrypt
}

func (e *EncryptedKey) parse(r io.Reader) (err os.Error) {
	var buf [10]byte
	_, err = readFull(r, buf[:])
	if err != nil {
		return
	}
	if buf[0] != 3 {
		return error.UnsupportedError("unknown EncryptedKey version " + strconv.Itoa(int(buf[0])))
	}
	e.KeyId = binary.BigEndian.Uint64(buf[1:9])
	e.Algo = PublicKeyAlgorithm(buf[9])
	if e.Algo == PubKeyAlgoRSA || e.Algo == PubKeyAlgoRSAEncryptOnly {
		e.Encrypted, _, err = readMPI(r)
	}
	_, err = consumeAll(r)
	return
}

// DecryptRSA decrypts an RSA encrypted session key with the given private key.
func (e *EncryptedKey) DecryptRSA(priv *rsa.PrivateKey) (err os.Error) {
	if e.Algo != PubKeyAlgoRSA && e.Algo != PubKeyAlgoRSAEncryptOnly {
		return error.InvalidArgumentError("EncryptedKey not RSA encrypted")
	}
	b, err := rsa.DecryptPKCS1v15(rand.Reader, priv, e.Encrypted)
	if err != nil {
		return
	}
	e.CipherFunc = CipherFunction(b[0])
	e.Key = b[1 : len(b)-2]
	expectedChecksum := uint16(b[len(b)-2])<<8 | uint16(b[len(b)-1])
	var checksum uint16
	for _, v := range e.Key {
		checksum += uint16(v)
	}
	if checksum != expectedChecksum {
		return error.StructuralError("EncryptedKey checksum incorrect")
	}

	return
}
