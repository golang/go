// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x509

// RFC 1423 describes the encryption of PEM blocks. The algorithm used to
// generate a key from the password was derived by looking at the OpenSSL
// implementation.

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/des"
	"crypto/md5"
	"encoding/hex"
	"encoding/pem"
	"errors"
	"strings"
)

// rfc1423Algos represents how to create a block cipher for a decryption mode.
type rfc1423Algo struct {
	cipherFunc func([]byte) (cipher.Block, error)
	keySize    int
}

// rfc1423Algos is a mapping of encryption algorithm to an rfc1423Algo that can
// create block ciphers for that mode.
var rfc1423Algos = map[string]rfc1423Algo{
	"DES-CBC":      {des.NewCipher, 8},
	"DES-EDE3-CBC": {des.NewTripleDESCipher, 24},
	"AES-128-CBC":  {aes.NewCipher, 16},
	"AES-192-CBC":  {aes.NewCipher, 24},
	"AES-256-CBC":  {aes.NewCipher, 32},
}

// deriveKey uses a key derivation function to stretch the password into a key
// with the number of bits our cipher requires. This algorithm was derived from
// the OpenSSL source.
func (c rfc1423Algo) deriveKey(password, salt []byte) []byte {
	hash := md5.New()
	out := make([]byte, c.keySize)
	var digest []byte

	for i := 0; i < len(out); i += len(digest) {
		hash.Reset()
		hash.Write(digest)
		hash.Write(password)
		hash.Write(salt)
		digest = hash.Sum(digest[:0])
		copy(out[i:], digest)
	}

	return out
}

// IsEncryptedPEMBlock returns if the PEM block is password encrypted.
func IsEncryptedPEMBlock(b *pem.Block) bool {
	_, ok := b.Headers["DEK-Info"]
	return ok
}

// IncorrectPasswordError is returned when an incorrect password is detected.
var IncorrectPasswordError = errors.New("x509: decryption password incorrect")

// DecryptPEMBlock takes a password encrypted PEM block and the password used to
// encrypt it and returns a slice of decrypted DER encoded bytes. It inspects
// the DEK-Info header to determine the algorithm used for decryption. If no
// DEK-Info header is present, an error is returned. If an incorrect password
// is detected an IncorrectPasswordError is returned.
func DecryptPEMBlock(b *pem.Block, password []byte) ([]byte, error) {
	dek, ok := b.Headers["DEK-Info"]
	if !ok {
		return nil, errors.New("x509: no DEK-Info header in block")
	}

	idx := strings.Index(dek, ",")
	if idx == -1 {
		return nil, errors.New("x509: malformed DEK-Info header")
	}

	mode, hexIV := dek[:idx], dek[idx+1:]
	ciph, ok := rfc1423Algos[mode]
	if !ok {
		return nil, errors.New("x509: unknown encryption mode")
	}
	iv, err := hex.DecodeString(hexIV)
	if err != nil {
		return nil, err
	}
	if len(iv) < 8 {
		return nil, errors.New("x509: not enough bytes in IV")
	}

	// Based on the OpenSSL implementation. The salt is the first 8 bytes
	// of the initialization vector.
	key := ciph.deriveKey(password, iv[:8])
	block, err := ciph.cipherFunc(key)
	if err != nil {
		return nil, err
	}

	data := make([]byte, len(b.Bytes))
	dec := cipher.NewCBCDecrypter(block, iv)
	dec.CryptBlocks(data, b.Bytes)

	// Blocks are padded using a scheme where the last n bytes of padding are all
	// equal to n. It can pad from 1 to blocksize bytes inclusive. See RFC 1423.
	// For example:
	//	[x y z 2 2]
	//	[x y 7 7 7 7 7 7 7]
	// If we detect a bad padding, we assume it is an invalid password.
	dlen := len(data)
	blockSize := block.BlockSize()
	if dlen == 0 || dlen%blockSize != 0 {
		return nil, errors.New("x509: invalid padding")
	}
	last := int(data[dlen-1])
	if dlen < last {
		return nil, IncorrectPasswordError
	}
	if last == 0 || last > blockSize {
		return nil, IncorrectPasswordError
	}
	for _, val := range data[dlen-last:] {
		if int(val) != last {
			return nil, IncorrectPasswordError
		}
	}
	return data[:dlen-last], nil
}
