// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package openpgp

import (
	"crypto"
	"crypto/openpgp/armor"
	"crypto/openpgp/error"
	"crypto/openpgp/packet"
	"crypto/rand"
	_ "crypto/sha256"
	"io"
	"os"
	"time"
)

// DetachSign signs message with the private key from signer (which must
// already have been decrypted) and writes the signature to w.
func DetachSign(w io.Writer, signer *Entity, message io.Reader) os.Error {
	return detachSign(w, signer, message, packet.SigTypeBinary)
}

// ArmoredDetachSign signs message with the private key from signer (which
// must already have been decrypted) and writes an armored signature to w.
func ArmoredDetachSign(w io.Writer, signer *Entity, message io.Reader) (err os.Error) {
	return armoredDetachSign(w, signer, message, packet.SigTypeBinary)
}

// DetachSignText signs message (after canonicalising the line endings) with
// the private key from signer (which must already have been decrypted) and
// writes the signature to w.
func DetachSignText(w io.Writer, signer *Entity, message io.Reader) os.Error {
	return detachSign(w, signer, message, packet.SigTypeText)
}

// ArmoredDetachSignText signs message (after canonicalising the line endings)
// with the private key from signer (which must already have been decrypted)
// and writes an armored signature to w.
func ArmoredDetachSignText(w io.Writer, signer *Entity, message io.Reader) os.Error {
	return armoredDetachSign(w, signer, message, packet.SigTypeText)
}

func armoredDetachSign(w io.Writer, signer *Entity, message io.Reader, sigType packet.SignatureType) (err os.Error) {
	out, err := armor.Encode(w, SignatureType, nil)
	if err != nil {
		return
	}
	err = detachSign(out, signer, message, sigType)
	if err != nil {
		return
	}
	return out.Close()
}

func detachSign(w io.Writer, signer *Entity, message io.Reader, sigType packet.SignatureType) (err os.Error) {
	if signer.PrivateKey == nil {
		return error.InvalidArgumentError("signing key doesn't have a private key")
	}
	if signer.PrivateKey.Encrypted {
		return error.InvalidArgumentError("signing key is encrypted")
	}

	sig := new(packet.Signature)
	sig.SigType = sigType
	sig.PubKeyAlgo = signer.PrivateKey.PubKeyAlgo
	sig.Hash = crypto.SHA256
	sig.CreationTime = uint32(time.Seconds())
	sig.IssuerKeyId = &signer.PrivateKey.KeyId

	h, wrappedHash, err := hashForSignature(sig.Hash, sig.SigType)
	if err != nil {
		return
	}
	io.Copy(wrappedHash, message)

	err = sig.Sign(h, signer.PrivateKey)
	if err != nil {
		return
	}

	return sig.Serialize(w)
}

// FileHints contains metadata about encrypted files. This metadata is, itself,
// encrypted.
type FileHints struct {
	// IsBinary can be set to hint that the contents are binary data.
	IsBinary bool
	// FileName hints at the name of the file that should be written. It's
	// truncated to 255 bytes if longer. It may be empty to suggest that the
	// file should not be written to disk. It may be equal to "_CONSOLE" to
	// suggest the data should not be written to disk.
	FileName string
	// EpochSeconds contains the modification time of the file, or 0 if not applicable.
	EpochSeconds uint32
}

// SymmetricallyEncrypt acts like gpg -c: it encrypts a file with a passphrase.
// The resulting WriteCloser MUST be closed after the contents of the file have
// been written.
func SymmetricallyEncrypt(ciphertext io.Writer, passphrase []byte, hints *FileHints) (plaintext io.WriteCloser, err os.Error) {
	if hints == nil {
		hints = &FileHints{}
	}

	key, err := packet.SerializeSymmetricKeyEncrypted(ciphertext, rand.Reader, passphrase, packet.CipherAES128)
	if err != nil {
		return
	}
	w, err := packet.SerializeSymmetricallyEncrypted(ciphertext, packet.CipherAES128, key)
	if err != nil {
		return
	}
	return packet.SerializeLiteral(w, hints.IsBinary, hints.FileName, hints.EpochSeconds)
}
