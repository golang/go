// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packet

import (
	"crypto/openpgp/elgamal"
	"crypto/openpgp/errors"
	"crypto/rand"
	"crypto/rsa"
	"encoding/binary"
	"io"
	"math/big"
	"strconv"
)

const encryptedKeyVersion = 3

// EncryptedKey represents a public-key encrypted session key. See RFC 4880,
// section 5.1.
type EncryptedKey struct {
	KeyId      uint64
	Algo       PublicKeyAlgorithm
	CipherFunc CipherFunction // only valid after a successful Decrypt
	Key        []byte         // only valid after a successful Decrypt

	encryptedMPI1, encryptedMPI2 []byte
}

func (e *EncryptedKey) parse(r io.Reader) (err error) {
	var buf [10]byte
	_, err = readFull(r, buf[:])
	if err != nil {
		return
	}
	if buf[0] != encryptedKeyVersion {
		return errors.UnsupportedError("unknown EncryptedKey version " + strconv.Itoa(int(buf[0])))
	}
	e.KeyId = binary.BigEndian.Uint64(buf[1:9])
	e.Algo = PublicKeyAlgorithm(buf[9])
	switch e.Algo {
	case PubKeyAlgoRSA, PubKeyAlgoRSAEncryptOnly:
		e.encryptedMPI1, _, err = readMPI(r)
	case PubKeyAlgoElGamal:
		e.encryptedMPI1, _, err = readMPI(r)
		if err != nil {
			return
		}
		e.encryptedMPI2, _, err = readMPI(r)
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

// Decrypt decrypts an encrypted session key with the given private key. The
// private key must have been decrypted first.
func (e *EncryptedKey) Decrypt(priv *PrivateKey) error {
	var err error
	var b []byte

	// TODO(agl): use session key decryption routines here to avoid
	// padding oracle attacks.
	switch priv.PubKeyAlgo {
	case PubKeyAlgoRSA, PubKeyAlgoRSAEncryptOnly:
		b, err = rsa.DecryptPKCS1v15(rand.Reader, priv.PrivateKey.(*rsa.PrivateKey), e.encryptedMPI1)
	case PubKeyAlgoElGamal:
		c1 := new(big.Int).SetBytes(e.encryptedMPI1)
		c2 := new(big.Int).SetBytes(e.encryptedMPI2)
		b, err = elgamal.Decrypt(priv.PrivateKey.(*elgamal.PrivateKey), c1, c2)
	default:
		err = errors.InvalidArgumentError("cannot decrypted encrypted session key with private key of type " + strconv.Itoa(int(priv.PubKeyAlgo)))
	}

	if err != nil {
		return err
	}

	e.CipherFunc = CipherFunction(b[0])
	e.Key = b[1 : len(b)-2]
	expectedChecksum := uint16(b[len(b)-2])<<8 | uint16(b[len(b)-1])
	checksum := checksumKeyMaterial(e.Key)
	if checksum != expectedChecksum {
		return errors.StructuralError("EncryptedKey checksum incorrect")
	}

	return nil
}

// SerializeEncryptedKey serializes an encrypted key packet to w that contains
// key, encrypted to pub.
func SerializeEncryptedKey(w io.Writer, rand io.Reader, pub *PublicKey, cipherFunc CipherFunction, key []byte) error {
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
	case PubKeyAlgoElGamal:
		return serializeEncryptedKeyElGamal(w, rand, buf, pub.PublicKey.(*elgamal.PublicKey), keyBlock)
	case PubKeyAlgoDSA, PubKeyAlgoRSASignOnly:
		return errors.InvalidArgumentError("cannot encrypt to public key of type " + strconv.Itoa(int(pub.PubKeyAlgo)))
	}

	return errors.UnsupportedError("encrypting a key to public key of type " + strconv.Itoa(int(pub.PubKeyAlgo)))
}

func serializeEncryptedKeyRSA(w io.Writer, rand io.Reader, header [10]byte, pub *rsa.PublicKey, keyBlock []byte) error {
	cipherText, err := rsa.EncryptPKCS1v15(rand, pub, keyBlock)
	if err != nil {
		return errors.InvalidArgumentError("RSA encryption failed: " + err.Error())
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

func serializeEncryptedKeyElGamal(w io.Writer, rand io.Reader, header [10]byte, pub *elgamal.PublicKey, keyBlock []byte) error {
	c1, c2, err := elgamal.Encrypt(rand, pub, keyBlock)
	if err != nil {
		return errors.InvalidArgumentError("ElGamal encryption failed: " + err.Error())
	}

	packetLen := 10 /* header length */
	packetLen += 2 /* mpi size */ + (c1.BitLen()+7)/8
	packetLen += 2 /* mpi size */ + (c2.BitLen()+7)/8

	err = serializeHeader(w, packetTypeEncryptedKey, packetLen)
	if err != nil {
		return err
	}
	_, err = w.Write(header[:])
	if err != nil {
		return err
	}
	err = writeBig(w, c1)
	if err != nil {
		return err
	}
	return writeBig(w, c2)
}
