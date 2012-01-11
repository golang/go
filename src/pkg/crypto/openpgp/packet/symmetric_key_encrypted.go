// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packet

import (
	"bytes"
	"crypto/cipher"
	"crypto/openpgp/errors"
	"crypto/openpgp/s2k"
	"io"
	"strconv"
)

// This is the largest session key that we'll support. Since no 512-bit cipher
// has even been seriously used, this is comfortably large.
const maxSessionKeySizeInBytes = 64

// SymmetricKeyEncrypted represents a passphrase protected session key. See RFC
// 4880, section 5.3.
type SymmetricKeyEncrypted struct {
	CipherFunc   CipherFunction
	Encrypted    bool
	Key          []byte // Empty unless Encrypted is false.
	s2k          func(out, in []byte)
	encryptedKey []byte
}

const symmetricKeyEncryptedVersion = 4

func (ske *SymmetricKeyEncrypted) parse(r io.Reader) (err error) {
	// RFC 4880, section 5.3.
	var buf [2]byte
	_, err = readFull(r, buf[:])
	if err != nil {
		return
	}
	if buf[0] != symmetricKeyEncryptedVersion {
		return errors.UnsupportedError("SymmetricKeyEncrypted version")
	}
	ske.CipherFunc = CipherFunction(buf[1])

	if ske.CipherFunc.KeySize() == 0 {
		return errors.UnsupportedError("unknown cipher: " + strconv.Itoa(int(buf[1])))
	}

	ske.s2k, err = s2k.Parse(r)
	if err != nil {
		return
	}

	encryptedKey := make([]byte, maxSessionKeySizeInBytes)
	// The session key may follow. We just have to try and read to find
	// out. If it exists then we limit it to maxSessionKeySizeInBytes.
	n, err := readFull(r, encryptedKey)
	if err != nil && err != io.ErrUnexpectedEOF {
		return
	}
	err = nil
	if n != 0 {
		if n == maxSessionKeySizeInBytes {
			return errors.UnsupportedError("oversized encrypted session key")
		}
		ske.encryptedKey = encryptedKey[:n]
	}

	ske.Encrypted = true

	return
}

// Decrypt attempts to decrypt an encrypted session key. If it returns nil,
// ske.Key will contain the session key.
func (ske *SymmetricKeyEncrypted) Decrypt(passphrase []byte) error {
	if !ske.Encrypted {
		return nil
	}

	key := make([]byte, ske.CipherFunc.KeySize())
	ske.s2k(key, passphrase)

	if len(ske.encryptedKey) == 0 {
		ske.Key = key
	} else {
		// the IV is all zeros
		iv := make([]byte, ske.CipherFunc.blockSize())
		c := cipher.NewCFBDecrypter(ske.CipherFunc.new(key), iv)
		c.XORKeyStream(ske.encryptedKey, ske.encryptedKey)
		ske.CipherFunc = CipherFunction(ske.encryptedKey[0])
		if ske.CipherFunc.blockSize() == 0 {
			return errors.UnsupportedError("unknown cipher: " + strconv.Itoa(int(ske.CipherFunc)))
		}
		ske.CipherFunc = CipherFunction(ske.encryptedKey[0])
		ske.Key = ske.encryptedKey[1:]
		if len(ske.Key)%ske.CipherFunc.blockSize() != 0 {
			ske.Key = nil
			return errors.StructuralError("length of decrypted key not a multiple of block size")
		}
	}

	ske.Encrypted = false
	return nil
}

// SerializeSymmetricKeyEncrypted serializes a symmetric key packet to w. The
// packet contains a random session key, encrypted by a key derived from the
// given passphrase. The session key is returned and must be passed to
// SerializeSymmetricallyEncrypted.
func SerializeSymmetricKeyEncrypted(w io.Writer, rand io.Reader, passphrase []byte, cipherFunc CipherFunction) (key []byte, err error) {
	keySize := cipherFunc.KeySize()
	if keySize == 0 {
		return nil, errors.UnsupportedError("unknown cipher: " + strconv.Itoa(int(cipherFunc)))
	}

	s2kBuf := new(bytes.Buffer)
	keyEncryptingKey := make([]byte, keySize)
	// s2k.Serialize salts and stretches the passphrase, and writes the
	// resulting key to keyEncryptingKey and the s2k descriptor to s2kBuf.
	err = s2k.Serialize(s2kBuf, keyEncryptingKey, rand, passphrase)
	if err != nil {
		return
	}
	s2kBytes := s2kBuf.Bytes()

	packetLength := 2 /* header */ + len(s2kBytes) + 1 /* cipher type */ + keySize
	err = serializeHeader(w, packetTypeSymmetricKeyEncrypted, packetLength)
	if err != nil {
		return
	}

	var buf [2]byte
	buf[0] = symmetricKeyEncryptedVersion
	buf[1] = byte(cipherFunc)
	_, err = w.Write(buf[:])
	if err != nil {
		return
	}
	_, err = w.Write(s2kBytes)
	if err != nil {
		return
	}

	sessionKey := make([]byte, keySize)
	_, err = io.ReadFull(rand, sessionKey)
	if err != nil {
		return
	}
	iv := make([]byte, cipherFunc.blockSize())
	c := cipher.NewCFBEncrypter(cipherFunc.new(keyEncryptingKey), iv)
	encryptedCipherAndKey := make([]byte, keySize+1)
	c.XORKeyStream(encryptedCipherAndKey, buf[1:])
	c.XORKeyStream(encryptedCipherAndKey[1:], sessionKey)
	_, err = w.Write(encryptedCipherAndKey)
	if err != nil {
		return
	}

	key = sessionKey
	return
}
