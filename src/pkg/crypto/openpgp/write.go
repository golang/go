// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package openpgp

import (
	"crypto"
	"crypto/dsa"
	"crypto/openpgp/armor"
	"crypto/openpgp/error"
	"crypto/openpgp/packet"
	"crypto/rsa"
	_ "crypto/sha256"
	"io"
	"os"
	"strconv"
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

	switch signer.PrivateKey.PubKeyAlgo {
	case packet.PubKeyAlgoRSA, packet.PubKeyAlgoRSASignOnly:
		priv := signer.PrivateKey.PrivateKey.(*rsa.PrivateKey)
		err = sig.SignRSA(h, priv)
	case packet.PubKeyAlgoDSA:
		priv := signer.PrivateKey.PrivateKey.(*dsa.PrivateKey)
		err = sig.SignDSA(h, priv)
	default:
		err = error.UnsupportedError("public key algorithm: " + strconv.Itoa(int(sig.PubKeyAlgo)))
	}

	if err != nil {
		return
	}

	return sig.Serialize(w)
}
