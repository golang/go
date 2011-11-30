// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package openpgp

import (
	"crypto"
	"crypto/openpgp/armor"
	error_ "crypto/openpgp/error"
	"crypto/openpgp/packet"
	"crypto/openpgp/s2k"
	"crypto/rand"
	_ "crypto/sha256"
	"hash"
	"io"
	"strconv"
	"time"
)

// DetachSign signs message with the private key from signer (which must
// already have been decrypted) and writes the signature to w.
func DetachSign(w io.Writer, signer *Entity, message io.Reader) error {
	return detachSign(w, signer, message, packet.SigTypeBinary)
}

// ArmoredDetachSign signs message with the private key from signer (which
// must already have been decrypted) and writes an armored signature to w.
func ArmoredDetachSign(w io.Writer, signer *Entity, message io.Reader) (err error) {
	return armoredDetachSign(w, signer, message, packet.SigTypeBinary)
}

// DetachSignText signs message (after canonicalising the line endings) with
// the private key from signer (which must already have been decrypted) and
// writes the signature to w.
func DetachSignText(w io.Writer, signer *Entity, message io.Reader) error {
	return detachSign(w, signer, message, packet.SigTypeText)
}

// ArmoredDetachSignText signs message (after canonicalising the line endings)
// with the private key from signer (which must already have been decrypted)
// and writes an armored signature to w.
func ArmoredDetachSignText(w io.Writer, signer *Entity, message io.Reader) error {
	return armoredDetachSign(w, signer, message, packet.SigTypeText)
}

func armoredDetachSign(w io.Writer, signer *Entity, message io.Reader, sigType packet.SignatureType) (err error) {
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

func detachSign(w io.Writer, signer *Entity, message io.Reader, sigType packet.SignatureType) (err error) {
	if signer.PrivateKey == nil {
		return error_.InvalidArgumentError("signing key doesn't have a private key")
	}
	if signer.PrivateKey.Encrypted {
		return error_.InvalidArgumentError("signing key is encrypted")
	}

	sig := new(packet.Signature)
	sig.SigType = sigType
	sig.PubKeyAlgo = signer.PrivateKey.PubKeyAlgo
	sig.Hash = crypto.SHA256
	sig.CreationTime = time.Now()
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
	// ModTime contains the modification time of the file, or the zero time if not applicable.
	ModTime time.Time
}

// SymmetricallyEncrypt acts like gpg -c: it encrypts a file with a passphrase.
// The resulting WriteCloser must be closed after the contents of the file have
// been written.
func SymmetricallyEncrypt(ciphertext io.Writer, passphrase []byte, hints *FileHints) (plaintext io.WriteCloser, err error) {
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
	var epochSeconds uint32
	if !hints.ModTime.IsZero() {
		epochSeconds = uint32(hints.ModTime.Unix())
	}
	return packet.SerializeLiteral(w, hints.IsBinary, hints.FileName, epochSeconds)
}

// intersectPreferences mutates and returns a prefix of a that contains only
// the values in the intersection of a and b. The order of a is preserved.
func intersectPreferences(a []uint8, b []uint8) (intersection []uint8) {
	var j int
	for _, v := range a {
		for _, v2 := range b {
			if v == v2 {
				a[j] = v
				j++
				break
			}
		}
	}

	return a[:j]
}

func hashToHashId(h crypto.Hash) uint8 {
	v, ok := s2k.HashToHashId(h)
	if !ok {
		panic("tried to convert unknown hash")
	}
	return v
}

// Encrypt encrypts a message to a number of recipients and, optionally, signs
// it. hints contains optional information, that is also encrypted, that aids
// the recipients in processing the message. The resulting WriteCloser must
// be closed after the contents of the file have been written.
func Encrypt(ciphertext io.Writer, to []*Entity, signed *Entity, hints *FileHints) (plaintext io.WriteCloser, err error) {
	var signer *packet.PrivateKey
	if signed != nil {
		signer = signed.signingKey().PrivateKey
		if signer == nil || signer.Encrypted {
			return nil, error_.InvalidArgumentError("signing key must be decrypted")
		}
	}

	// These are the possible ciphers that we'll use for the message.
	candidateCiphers := []uint8{
		uint8(packet.CipherAES128),
		uint8(packet.CipherAES256),
		uint8(packet.CipherCAST5),
	}
	// These are the possible hash functions that we'll use for the signature.
	candidateHashes := []uint8{
		hashToHashId(crypto.SHA256),
		hashToHashId(crypto.SHA512),
		hashToHashId(crypto.SHA1),
		hashToHashId(crypto.RIPEMD160),
	}
	// In the event that a recipient doesn't specify any supported ciphers
	// or hash functions, these are the ones that we assume that every
	// implementation supports.
	defaultCiphers := candidateCiphers[len(candidateCiphers)-1:]
	defaultHashes := candidateHashes[len(candidateHashes)-1:]

	encryptKeys := make([]Key, len(to))
	for i := range to {
		encryptKeys[i] = to[i].encryptionKey()
		if encryptKeys[i].PublicKey == nil {
			return nil, error_.InvalidArgumentError("cannot encrypt a message to key id " + strconv.Uitob64(to[i].PrimaryKey.KeyId, 16) + " because it has no encryption keys")
		}

		sig := to[i].primaryIdentity().SelfSignature

		preferredSymmetric := sig.PreferredSymmetric
		if len(preferredSymmetric) == 0 {
			preferredSymmetric = defaultCiphers
		}
		preferredHashes := sig.PreferredHash
		if len(preferredHashes) == 0 {
			preferredHashes = defaultHashes
		}
		candidateCiphers = intersectPreferences(candidateCiphers, preferredSymmetric)
		candidateHashes = intersectPreferences(candidateHashes, preferredHashes)
	}

	if len(candidateCiphers) == 0 || len(candidateHashes) == 0 {
		return nil, error_.InvalidArgumentError("cannot encrypt because recipient set shares no common algorithms")
	}

	cipher := packet.CipherFunction(candidateCiphers[0])
	hash, _ := s2k.HashIdToHash(candidateHashes[0])
	symKey := make([]byte, cipher.KeySize())
	if _, err := io.ReadFull(rand.Reader, symKey); err != nil {
		return nil, err
	}

	for _, key := range encryptKeys {
		if err := packet.SerializeEncryptedKey(ciphertext, rand.Reader, key.PublicKey, cipher, symKey); err != nil {
			return nil, err
		}
	}

	encryptedData, err := packet.SerializeSymmetricallyEncrypted(ciphertext, cipher, symKey)
	if err != nil {
		return
	}

	if signer != nil {
		ops := &packet.OnePassSignature{
			SigType:    packet.SigTypeBinary,
			Hash:       hash,
			PubKeyAlgo: signer.PubKeyAlgo,
			KeyId:      signer.KeyId,
			IsLast:     true,
		}
		if err := ops.Serialize(encryptedData); err != nil {
			return nil, err
		}
	}

	if hints == nil {
		hints = &FileHints{}
	}

	w := encryptedData
	if signer != nil {
		// If we need to write a signature packet after the literal
		// data then we need to stop literalData from closing
		// encryptedData.
		w = noOpCloser{encryptedData}

	}
	var epochSeconds uint32
	if !hints.ModTime.IsZero() {
		epochSeconds = uint32(hints.ModTime.Unix())
	}
	literalData, err := packet.SerializeLiteral(w, hints.IsBinary, hints.FileName, epochSeconds)
	if err != nil {
		return nil, err
	}

	if signer != nil {
		return signatureWriter{encryptedData, literalData, hash, hash.New(), signer}, nil
	}
	return literalData, nil
}

// signatureWriter hashes the contents of a message while passing it along to
// literalData. When closed, it closes literalData, writes a signature packet
// to encryptedData and then also closes encryptedData.
type signatureWriter struct {
	encryptedData io.WriteCloser
	literalData   io.WriteCloser
	hashType      crypto.Hash
	h             hash.Hash
	signer        *packet.PrivateKey
}

func (s signatureWriter) Write(data []byte) (int, error) {
	s.h.Write(data)
	return s.literalData.Write(data)
}

func (s signatureWriter) Close() error {
	sig := &packet.Signature{
		SigType:      packet.SigTypeBinary,
		PubKeyAlgo:   s.signer.PubKeyAlgo,
		Hash:         s.hashType,
		CreationTime: time.Now(),
		IssuerKeyId:  &s.signer.KeyId,
	}

	if err := sig.Sign(s.h, s.signer); err != nil {
		return err
	}
	if err := s.literalData.Close(); err != nil {
		return err
	}
	if err := sig.Serialize(s.encryptedData); err != nil {
		return err
	}
	return s.encryptedData.Close()
}

// noOpCloser is like an ioutil.NopCloser, but for an io.Writer.
// TODO: we have two of these in OpenPGP packages alone. This probably needs
// to be promoted somewhere more common.
type noOpCloser struct {
	w io.Writer
}

func (c noOpCloser) Write(data []byte) (n int, err error) {
	return c.w.Write(data)
}

func (c noOpCloser) Close() error {
	return nil
}
