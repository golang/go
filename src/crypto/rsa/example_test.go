// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rsa

import (
	"crypto"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
)

// RSA is able to encrypt only a very limited amount of data. In order
// to encrypt reasonable amounts of data a hybrid scheme is commonly
// used: RSA is used to encrypt a key for a symmetric primitive like
// AES-GCM.
//
// Before encrypting, data is “padded” by embedding it in a known
// structure. This is done for a number of reasons, but the most
// obvious is to ensure that the value is large enough that the
// exponentiation is larger than the modulus. (Otherwise it could be
// decrypted with a square-root.)
//
// In these designs, when using PKCS#1 v1.5, it's vitally important to
// avoid disclosing whether the received RSA message was well-formed
// (that is, whether the result of decrypting is a correctly padded
// message) because this leaks secret information.
// DecryptPKCS1v15SessionKey is designed for this situation and copies
// the decrypted, symmetric key (if well-formed) in constant-time over
// a buffer that contains a random key. Thus, if the RSA result isn't
// well-formed, the implementation uses a random key in constant time.
func ExampleDecryptPKCS1v15SessionKey() {
	// crypto/rand.Reader is a good source of entropy for blinding the RSA
	// operation.
	rng := rand.Reader

	// The hybrid scheme should use at least a 16-byte symmetric key. Here
	// we read the random key that will be used if the RSA decryption isn't
	// well-formed.
	key := make([]byte, 32)
	if _, err := io.ReadFull(rng, key); err != nil {
		panic("RNG failure")
	}

	rsaCiphertext, _ := hex.DecodeString("aabbccddeeff")

	if err := DecryptPKCS1v15SessionKey(rng, rsaPrivateKey, rsaCiphertext, key); err != nil {
		// Any errors that result will be “public” – meaning that they
		// can be determined without any secret information. (For
		// instance, if the length of key is impossible given the RSA
		// public key.)
		fmt.Fprintf(os.Stderr, "Error from RSA decryption: %s\n", err)
		return
	}

	// Given the resulting key, a symmetric scheme can be used to decrypt a
	// larger ciphertext.
	block, err := aes.NewCipher(key)
	if err != nil {
		panic("aes.NewCipher failed: " + err.Error())
	}

	// Since the key is random, using a fixed nonce is acceptable as the
	// (key, nonce) pair will still be unique, as required.
	var zeroNonce [12]byte
	aead, err := cipher.NewGCM(block)
	if err != nil {
		panic("cipher.NewGCM failed: " + err.Error())
	}
	ciphertext, _ := hex.DecodeString("00112233445566")
	plaintext, err := aead.Open(nil, zeroNonce[:], ciphertext, nil)
	if err != nil {
		// The RSA ciphertext was badly formed; the decryption will
		// fail here because the AES-GCM key will be incorrect.
		fmt.Fprintf(os.Stderr, "Error decrypting: %s\n", err)
		return
	}

	fmt.Printf("Plaintext: %s\n", string(plaintext))
}

func ExampleSignPKCS1v15() {
	// crypto/rand.Reader is a good source of entropy for blinding the RSA
	// operation.
	rng := rand.Reader

	message := []byte("message to be signed")

	// Only small messages can be signed directly; thus the hash of a
	// message, rather than the message itself, is signed. This requires
	// that the hash function be collision resistant. SHA-256 is the
	// least-strong hash function that should be used for this at the time
	// of writing (2016).
	hashed := sha256.Sum256(message)

	signature, err := SignPKCS1v15(rng, rsaPrivateKey, crypto.SHA256, hashed[:])
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error from signing: %s\n", err)
		return
	}

	fmt.Printf("Signature: %x\n", signature)
}

func ExampleVerifyPKCS1v15() {
	message := []byte("message to be signed")
	signature, _ := hex.DecodeString("ad2766728615cc7a746cc553916380ca7bfa4f8983b990913bc69eb0556539a350ff0f8fe65ddfd3ebe91fe1c299c2fac135bc8c61e26be44ee259f2f80c1530")

	// Only small messages can be signed directly; thus the hash of a
	// message, rather than the message itself, is signed. This requires
	// that the hash function be collision resistant. SHA-256 is the
	// least-strong hash function that should be used for this at the time
	// of writing (2016).
	hashed := sha256.Sum256(message)

	err := VerifyPKCS1v15(&rsaPrivateKey.PublicKey, crypto.SHA256, hashed[:], signature)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error from verification: %s\n", err)
		return
	}

	// signature is a valid signature of message from the public key.
}

func ExampleEncryptOAEP() {
	secretMessage := []byte("send reinforcements, we're going to advance")
	label := []byte("orders")

	// crypto/rand.Reader is a good source of entropy for randomizing the
	// encryption function.
	rng := rand.Reader

	ciphertext, err := EncryptOAEP(sha256.New(), rng, &test2048Key.PublicKey, secretMessage, label)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error from encryption: %s\n", err)
		return
	}

	// Since encryption is a randomized function, ciphertext will be
	// different each time.
	fmt.Printf("Ciphertext: %x\n", ciphertext)
}

func ExampleDecryptOAEP() {
	ciphertext, _ := hex.DecodeString("4d1ee10e8f286390258c51a5e80802844c3e6358ad6690b7285218a7c7ed7fc3a4c7b950fbd04d4b0239cc060dcc7065ca6f84c1756deb71ca5685cadbb82be025e16449b905c568a19c088a1abfad54bf7ecc67a7df39943ec511091a34c0f2348d04e058fcff4d55644de3cd1d580791d4524b92f3e91695582e6e340a1c50b6c6d78e80b4e42c5b4d45e479b492de42bbd39cc642ebb80226bb5200020d501b24a37bcc2ec7f34e596b4fd6b063de4858dbf5a4e3dd18e262eda0ec2d19dbd8e890d672b63d368768360b20c0b6b8592a438fa275e5fa7f60bef0dd39673fd3989cc54d2cb80c08fcd19dacbc265ee1c6014616b0e04ea0328c2a04e73460")
	label := []byte("orders")

	// crypto/rand.Reader is a good source of entropy for blinding the RSA
	// operation.
	rng := rand.Reader

	plaintext, err := DecryptOAEP(sha256.New(), rng, test2048Key, ciphertext, label)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error from decryption: %s\n", err)
		return
	}

	fmt.Printf("Plaintext: %s\n", string(plaintext))

	// Remember that encryption only provides confidentiality. The
	// ciphertext should be signed before authenticity is assumed and, even
	// then, consider that messages might be reordered.
}
