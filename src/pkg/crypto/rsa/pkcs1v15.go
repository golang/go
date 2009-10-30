// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rsa

import (
	"bytes";
	big "gmp";
	"io";
	"os";
)

// This file implements encryption and decryption using PKCS#1 v1.5 padding.

// EncryptPKCS1v15 encrypts the given message with RSA and the padding scheme from PKCS#1 v1.5.
// The message must be no longer than the length of the public modulus minus 11 bytes.
// WARNING: use of this function to encrypt plaintexts other than session keys
// is dangerous. Use RSA OAEP in new protocols.
func EncryptPKCS1v15(rand io.Reader, pub *PublicKey, msg []byte) (out []byte, err os.Error) {
	k := (pub.N.Len() + 7)/8;
	if len(msg) > k-11 {
		err = MessageTooLongError{};
		return;
	}

	// EM = 0x02 || PS || 0x00 || M
	em := make([]byte, k-1);
	em[0] = 2;
	ps, mm := em[1:len(em)-len(msg)-1], em[len(em)-len(msg):len(em)];
	err = nonZeroRandomBytes(ps, rand);
	if err != nil {
		return;
	}
	em[len(em)-len(msg)-1] = 0;
	bytes.Copy(mm, msg);

	m := new(big.Int).SetBytes(em);
	c := encrypt(new(big.Int), pub, m);
	out = c.Bytes();
	return;
}

// DecryptPKCS1v15 decrypts a plaintext using RSA and the padding scheme from PKCS#1 v1.5.
// If rand != nil, it uses RSA blinding to avoid timing side-channel attacks.
func DecryptPKCS1v15(rand io.Reader, priv *PrivateKey, ciphertext []byte) (out []byte, err os.Error) {
	valid, out, err := decryptPKCS1v15(rand, priv, ciphertext);
	if err == nil && valid == 0 {
		err = DecryptionError{};
	}

	return;
}

// DecryptPKCS1v15SessionKey decrypts a session key using RSA and the padding scheme from PKCS#1 v1.5.
// If rand != nil, it uses RSA blinding to avoid timing side-channel attacks.
// It returns an error if the ciphertext is the wrong length or if the
// ciphertext is greater than the public modulus. Otherwise, no error is
// returned. If the padding is valid, the resulting plaintext message is copied
// into key. Otherwise, key is unchanged. These alternatives occur in constant
// time. It is intended that the user of this function generate a random
// session key beforehand and continue the protocol with the resulting value.
// This will remove any possibility that an attacker can learn any information
// about the plaintext.
// See ``Chosen Ciphertext Attacks Against Protocols Based on the RSA
// Encryption Standard PKCS #1'', Daniel Bleichenbacher, Advances in Cryptology
// (Crypto '98),
func DecryptPKCS1v15SessionKey(rand io.Reader, priv *PrivateKey, ciphertext []byte, key []byte) (err os.Error) {
	k := (priv.N.Len() + 7)/8;
	if k-(len(key)+3+8) < 0 {
		err = DecryptionError{};
		return;
	}

	valid, msg, err := decryptPKCS1v15(rand, priv, ciphertext);
	if err != nil {
		return;
	}

	valid &= constantTimeEq(int32(len(msg)), int32(len(key)));
	constantTimeCopy(valid, key, msg);
	return;
}

func decryptPKCS1v15(rand io.Reader, priv *PrivateKey, ciphertext []byte) (valid int, msg []byte, err os.Error) {
	k := (priv.N.Len() + 7)/8;
	if k < 11 {
		err = DecryptionError{};
		return;
	}

	c := new(big.Int).SetBytes(ciphertext);
	m, err := decrypt(rand, priv, c);
	if err != nil {
		return;
	}

	em := leftPad(m.Bytes(), k);
	firstByteIsZero := constantTimeByteEq(em[0], 0);
	secondByteIsTwo := constantTimeByteEq(em[1], 2);

	// The remainder of the plaintext must be a string of non-zero random
	// octets, followed by a 0, followed by the message.
	//   lookingForIndex: 1 iff we are still looking for the zero.
	//   index: the offset of the first zero byte.
	var lookingForIndex, index int;
	lookingForIndex = 1;

	for i := 2; i < len(em); i++ {
		equals0 := constantTimeByteEq(em[i], 0);
		index = constantTimeSelect(lookingForIndex & equals0, i, index);
		lookingForIndex = constantTimeSelect(equals0, 0, lookingForIndex);
	}

	valid = firstByteIsZero & secondByteIsTwo & (^lookingForIndex & 1);
	msg = em[index+1 : len(em)];
	return;
}

// nonZeroRandomBytes fills the given slice with non-zero random octets.
func nonZeroRandomBytes(s []byte, rand io.Reader) (err os.Error) {
	_, err = io.ReadFull(rand, s);
	if err != nil {
		return;
	}

	for i := 0; i < len(s); i++ {
		for s[i] == 0 {
			_, err = rand.Read(s[i:i+1]);
			if err != nil {
				return;
			}
		}
	}

	return;
}
