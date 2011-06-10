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

const encryptedKeyVersion = 3

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
	if buf[0] != encryptedKeyVersion {
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

func checksumKeyMaterial(key []byte) uint16 {
	var checksum uint16
	for _, v := range key {
		checksum += uint16(v)
	}
	return checksum
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
	checksum := checksumKeyMaterial(e.Key)
	if checksum != expectedChecksum {
		return error.StructuralError("EncryptedKey checksum incorrect")
	}

	return
}

// SerializeEncryptedKey serializes an encrypted key packet to w that contains
// key, encrypted to pub.
func SerializeEncryptedKey(w io.Writer, rand io.Reader, pub *PublicKey, cipherFunc CipherFunction, key []byte) os.Error {
	var buf [10]byte
	buf[0] = encryptedKeyVersion
	binary.BigEndian.PutUint64(buf[1:9], pub.KeyId)
	buf[9] = byte(pub.PubKeyAlgo)

	keyBlock := make([]byte, 1 /* cipher type */ +len(key)+2 /* checksum */ )
	keyBlock[0] = byte(cipherFunc)
	copy(keyBlock[1:], key)
	checksum := checksumKeyMaterial(key)
	keyBlock[1+len(key)] = byte(checksum >> 8)
	keyBlock[1+len(key)+1] = byte(checksum)

	switch pub.PubKeyAlgo {
	case PubKeyAlgoRSA, PubKeyAlgoRSAEncryptOnly:
		return serializeEncryptedKeyRSA(w, rand, buf, pub.PublicKey.(*rsa.PublicKey), keyBlock)
	case PubKeyAlgoDSA, PubKeyAlgoRSASignOnly:
		return error.InvalidArgumentError("cannot encrypt to public key of type " + strconv.Itoa(int(pub.PubKeyAlgo)))
	}

	return error.UnsupportedError("encrypting a key to public key of type " + strconv.Itoa(int(pub.PubKeyAlgo)))
}

func serializeEncryptedKeyRSA(w io.Writer, rand io.Reader, header [10]byte, pub *rsa.PublicKey, keyBlock []byte) os.Error {
	cipherText, err := rsa.EncryptPKCS1v15(rand, pub, keyBlock)
	if err != nil {
		return error.InvalidArgumentError("RSA encryption failed: " + err.String())
	}

	packetLen := 10 /* header length */ + 2 /* mpi size */ + len(cipherText)

	err = serializeHeader(w, packetTypeEncryptedKey, packetLen)
	if err != nil {
		return err
	}
	_, err = w.Write(header[:])
	if err != nil {
		return err
	}
	return writeMPI(w, 8*uint16(len(cipherText)), cipherText)
}
