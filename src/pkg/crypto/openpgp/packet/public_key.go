// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packet

import (
	"crypto/dsa"
	"crypto/openpgp/elgamal"
	"crypto/openpgp/errors"
	"crypto/rsa"
	"crypto/sha1"
	"encoding/binary"
	"fmt"
	"hash"
	"io"
	"math/big"
	"strconv"
	"time"
)

// PublicKey represents an OpenPGP public key. See RFC 4880, section 5.5.2.
type PublicKey struct {
	CreationTime time.Time
	PubKeyAlgo   PublicKeyAlgorithm
	PublicKey    interface{} // Either a *rsa.PublicKey or *dsa.PublicKey
	Fingerprint  [20]byte
	KeyId        uint64
	IsSubkey     bool

	n, e, p, q, g, y parsedMPI
}

func fromBig(n *big.Int) parsedMPI {
	return parsedMPI{
		bytes:     n.Bytes(),
		bitLength: uint16(n.BitLen()),
	}
}

// NewRSAPublicKey returns a PublicKey that wraps the given rsa.PublicKey.
func NewRSAPublicKey(creationTime time.Time, pub *rsa.PublicKey) *PublicKey {
	pk := &PublicKey{
		CreationTime: creationTime,
		PubKeyAlgo:   PubKeyAlgoRSA,
		PublicKey:    pub,
		n:            fromBig(pub.N),
		e:            fromBig(big.NewInt(int64(pub.E))),
	}

	pk.setFingerPrintAndKeyId()
	return pk
}

// NewDSAPublicKey returns a PublicKey that wraps the given rsa.PublicKey.
func NewDSAPublicKey(creationTime time.Time, pub *dsa.PublicKey) *PublicKey {
	pk := &PublicKey{
		CreationTime: creationTime,
		PubKeyAlgo:   PubKeyAlgoDSA,
		PublicKey:    pub,
		p:            fromBig(pub.P),
		q:            fromBig(pub.Q),
		g:            fromBig(pub.G),
		y:            fromBig(pub.Y),
	}

	pk.setFingerPrintAndKeyId()
	return pk
}

func (pk *PublicKey) parse(r io.Reader) (err error) {
	// RFC 4880, section 5.5.2
	var buf [6]byte
	_, err = readFull(r, buf[:])
	if err != nil {
		return
	}
	if buf[0] != 4 {
		return errors.UnsupportedError("public key version")
	}
	pk.CreationTime = time.Unix(int64(uint32(buf[1])<<24|uint32(buf[2])<<16|uint32(buf[3])<<8|uint32(buf[4])), 0)
	pk.PubKeyAlgo = PublicKeyAlgorithm(buf[5])
	switch pk.PubKeyAlgo {
	case PubKeyAlgoRSA, PubKeyAlgoRSAEncryptOnly, PubKeyAlgoRSASignOnly:
		err = pk.parseRSA(r)
	case PubKeyAlgoDSA:
		err = pk.parseDSA(r)
	case PubKeyAlgoElGamal:
		err = pk.parseElGamal(r)
	default:
		err = errors.UnsupportedError("public key type: " + strconv.Itoa(int(pk.PubKeyAlgo)))
	}
	if err != nil {
		return
	}

	pk.setFingerPrintAndKeyId()
	return
}

func (pk *PublicKey) setFingerPrintAndKeyId() {
	// RFC 4880, section 12.2
	fingerPrint := sha1.New()
	pk.SerializeSignaturePrefix(fingerPrint)
	pk.serializeWithoutHeaders(fingerPrint)
	copy(pk.Fingerprint[:], fingerPrint.Sum(nil))
	pk.KeyId = binary.BigEndian.Uint64(pk.Fingerprint[12:20])
}

// parseRSA parses RSA public key material from the given Reader. See RFC 4880,
// section 5.5.2.
func (pk *PublicKey) parseRSA(r io.Reader) (err error) {
	pk.n.bytes, pk.n.bitLength, err = readMPI(r)
	if err != nil {
		return
	}
	pk.e.bytes, pk.e.bitLength, err = readMPI(r)
	if err != nil {
		return
	}

	if len(pk.e.bytes) > 3 {
		err = errors.UnsupportedError("large public exponent")
		return
	}
	rsa := &rsa.PublicKey{
		N: new(big.Int).SetBytes(pk.n.bytes),
		E: 0,
	}
	for i := 0; i < len(pk.e.bytes); i++ {
		rsa.E <<= 8
		rsa.E |= int(pk.e.bytes[i])
	}
	pk.PublicKey = rsa
	return
}

// parseDSA parses DSA public key material from the given Reader. See RFC 4880,
// section 5.5.2.
func (pk *PublicKey) parseDSA(r io.Reader) (err error) {
	pk.p.bytes, pk.p.bitLength, err = readMPI(r)
	if err != nil {
		return
	}
	pk.q.bytes, pk.q.bitLength, err = readMPI(r)
	if err != nil {
		return
	}
	pk.g.bytes, pk.g.bitLength, err = readMPI(r)
	if err != nil {
		return
	}
	pk.y.bytes, pk.y.bitLength, err = readMPI(r)
	if err != nil {
		return
	}

	dsa := new(dsa.PublicKey)
	dsa.P = new(big.Int).SetBytes(pk.p.bytes)
	dsa.Q = new(big.Int).SetBytes(pk.q.bytes)
	dsa.G = new(big.Int).SetBytes(pk.g.bytes)
	dsa.Y = new(big.Int).SetBytes(pk.y.bytes)
	pk.PublicKey = dsa
	return
}

// parseElGamal parses ElGamal public key material from the given Reader. See
// RFC 4880, section 5.5.2.
func (pk *PublicKey) parseElGamal(r io.Reader) (err error) {
	pk.p.bytes, pk.p.bitLength, err = readMPI(r)
	if err != nil {
		return
	}
	pk.g.bytes, pk.g.bitLength, err = readMPI(r)
	if err != nil {
		return
	}
	pk.y.bytes, pk.y.bitLength, err = readMPI(r)
	if err != nil {
		return
	}

	elgamal := new(elgamal.PublicKey)
	elgamal.P = new(big.Int).SetBytes(pk.p.bytes)
	elgamal.G = new(big.Int).SetBytes(pk.g.bytes)
	elgamal.Y = new(big.Int).SetBytes(pk.y.bytes)
	pk.PublicKey = elgamal
	return
}

// SerializeSignaturePrefix writes the prefix for this public key to the given Writer.
// The prefix is used when calculating a signature over this public key. See
// RFC 4880, section 5.2.4.
func (pk *PublicKey) SerializeSignaturePrefix(h hash.Hash) {
	var pLength uint16
	switch pk.PubKeyAlgo {
	case PubKeyAlgoRSA, PubKeyAlgoRSAEncryptOnly, PubKeyAlgoRSASignOnly:
		pLength += 2 + uint16(len(pk.n.bytes))
		pLength += 2 + uint16(len(pk.e.bytes))
	case PubKeyAlgoDSA:
		pLength += 2 + uint16(len(pk.p.bytes))
		pLength += 2 + uint16(len(pk.q.bytes))
		pLength += 2 + uint16(len(pk.g.bytes))
		pLength += 2 + uint16(len(pk.y.bytes))
	case PubKeyAlgoElGamal:
		pLength += 2 + uint16(len(pk.p.bytes))
		pLength += 2 + uint16(len(pk.g.bytes))
		pLength += 2 + uint16(len(pk.y.bytes))
	default:
		panic("unknown public key algorithm")
	}
	pLength += 6
	h.Write([]byte{0x99, byte(pLength >> 8), byte(pLength)})
	return
}

func (pk *PublicKey) Serialize(w io.Writer) (err error) {
	length := 6 // 6 byte header

	switch pk.PubKeyAlgo {
	case PubKeyAlgoRSA, PubKeyAlgoRSAEncryptOnly, PubKeyAlgoRSASignOnly:
		length += 2 + len(pk.n.bytes)
		length += 2 + len(pk.e.bytes)
	case PubKeyAlgoDSA:
		length += 2 + len(pk.p.bytes)
		length += 2 + len(pk.q.bytes)
		length += 2 + len(pk.g.bytes)
		length += 2 + len(pk.y.bytes)
	case PubKeyAlgoElGamal:
		length += 2 + len(pk.p.bytes)
		length += 2 + len(pk.g.bytes)
		length += 2 + len(pk.y.bytes)
	default:
		panic("unknown public key algorithm")
	}

	packetType := packetTypePublicKey
	if pk.IsSubkey {
		packetType = packetTypePublicSubkey
	}
	err = serializeHeader(w, packetType, length)
	if err != nil {
		return
	}
	return pk.serializeWithoutHeaders(w)
}

// serializeWithoutHeaders marshals the PublicKey to w in the form of an
// OpenPGP public key packet, not including the packet header.
func (pk *PublicKey) serializeWithoutHeaders(w io.Writer) (err error) {
	var buf [6]byte
	buf[0] = 4
	t := uint32(pk.CreationTime.Unix())
	buf[1] = byte(t >> 24)
	buf[2] = byte(t >> 16)
	buf[3] = byte(t >> 8)
	buf[4] = byte(t)
	buf[5] = byte(pk.PubKeyAlgo)

	_, err = w.Write(buf[:])
	if err != nil {
		return
	}

	switch pk.PubKeyAlgo {
	case PubKeyAlgoRSA, PubKeyAlgoRSAEncryptOnly, PubKeyAlgoRSASignOnly:
		return writeMPIs(w, pk.n, pk.e)
	case PubKeyAlgoDSA:
		return writeMPIs(w, pk.p, pk.q, pk.g, pk.y)
	case PubKeyAlgoElGamal:
		return writeMPIs(w, pk.p, pk.g, pk.y)
	}
	return errors.InvalidArgumentError("bad public-key algorithm")
}

// CanSign returns true iff this public key can generate signatures
func (pk *PublicKey) CanSign() bool {
	return pk.PubKeyAlgo != PubKeyAlgoRSAEncryptOnly && pk.PubKeyAlgo != PubKeyAlgoElGamal
}

// VerifySignature returns nil iff sig is a valid signature, made by this
// public key, of the data hashed into signed. signed is mutated by this call.
func (pk *PublicKey) VerifySignature(signed hash.Hash, sig *Signature) (err error) {
	if !pk.CanSign() {
		return errors.InvalidArgumentError("public key cannot generate signatures")
	}

	signed.Write(sig.HashSuffix)
	hashBytes := signed.Sum(nil)

	if hashBytes[0] != sig.HashTag[0] || hashBytes[1] != sig.HashTag[1] {
		return errors.SignatureError("hash tag doesn't match")
	}

	if pk.PubKeyAlgo != sig.PubKeyAlgo {
		return errors.InvalidArgumentError("public key and signature use different algorithms")
	}

	switch pk.PubKeyAlgo {
	case PubKeyAlgoRSA, PubKeyAlgoRSASignOnly:
		rsaPublicKey, _ := pk.PublicKey.(*rsa.PublicKey)
		err = rsa.VerifyPKCS1v15(rsaPublicKey, sig.Hash, hashBytes, sig.RSASignature.bytes)
		if err != nil {
			return errors.SignatureError("RSA verification failure")
		}
		return nil
	case PubKeyAlgoDSA:
		dsaPublicKey, _ := pk.PublicKey.(*dsa.PublicKey)
		// Need to truncate hashBytes to match FIPS 186-3 section 4.6.
		subgroupSize := (dsaPublicKey.Q.BitLen() + 7) / 8
		if len(hashBytes) > subgroupSize {
			hashBytes = hashBytes[:subgroupSize]
		}
		if !dsa.Verify(dsaPublicKey, hashBytes, new(big.Int).SetBytes(sig.DSASigR.bytes), new(big.Int).SetBytes(sig.DSASigS.bytes)) {
			return errors.SignatureError("DSA verification failure")
		}
		return nil
	default:
		panic("shouldn't happen")
	}
	panic("unreachable")
}

// keySignatureHash returns a Hash of the message that needs to be signed for
// pk to assert a subkey relationship to signed.
func keySignatureHash(pk, signed *PublicKey, sig *Signature) (h hash.Hash, err error) {
	h = sig.Hash.New()
	if h == nil {
		return nil, errors.UnsupportedError("hash function")
	}

	// RFC 4880, section 5.2.4
	pk.SerializeSignaturePrefix(h)
	pk.serializeWithoutHeaders(h)
	signed.SerializeSignaturePrefix(h)
	signed.serializeWithoutHeaders(h)
	return
}

// VerifyKeySignature returns nil iff sig is a valid signature, made by this
// public key, of signed.
func (pk *PublicKey) VerifyKeySignature(signed *PublicKey, sig *Signature) (err error) {
	h, err := keySignatureHash(pk, signed, sig)
	if err != nil {
		return err
	}
	return pk.VerifySignature(h, sig)
}

// userIdSignatureHash returns a Hash of the message that needs to be signed
// to assert that pk is a valid key for id.
func userIdSignatureHash(id string, pk *PublicKey, sig *Signature) (h hash.Hash, err error) {
	h = sig.Hash.New()
	if h == nil {
		return nil, errors.UnsupportedError("hash function")
	}

	// RFC 4880, section 5.2.4
	pk.SerializeSignaturePrefix(h)
	pk.serializeWithoutHeaders(h)

	var buf [5]byte
	buf[0] = 0xb4
	buf[1] = byte(len(id) >> 24)
	buf[2] = byte(len(id) >> 16)
	buf[3] = byte(len(id) >> 8)
	buf[4] = byte(len(id))
	h.Write(buf[:])
	h.Write([]byte(id))

	return
}

// VerifyUserIdSignature returns nil iff sig is a valid signature, made by this
// public key, of id.
func (pk *PublicKey) VerifyUserIdSignature(id string, sig *Signature) (err error) {
	h, err := userIdSignatureHash(id, pk, sig)
	if err != nil {
		return err
	}
	return pk.VerifySignature(h, sig)
}

// KeyIdString returns the public key's fingerprint in capital hex
// (e.g. "6C7EE1B8621CC013").
func (pk *PublicKey) KeyIdString() string {
	return fmt.Sprintf("%X", pk.Fingerprint[12:20])
}

// KeyIdShortString returns the short form of public key's fingerprint
// in capital hex, as shown by gpg --list-keys (e.g. "621CC013").
func (pk *PublicKey) KeyIdShortString() string {
	return fmt.Sprintf("%X", pk.Fingerprint[16:20])
}

// A parsedMPI is used to store the contents of a big integer, along with the
// bit length that was specified in the original input. This allows the MPI to
// be reserialized exactly.
type parsedMPI struct {
	bytes     []byte
	bitLength uint16
}

// writeMPIs is a utility function for serializing several big integers to the
// given Writer.
func writeMPIs(w io.Writer, mpis ...parsedMPI) (err error) {
	for _, mpi := range mpis {
		err = writeMPI(w, mpi.bitLength, mpi.bytes)
		if err != nil {
			return
		}
	}
	return
}
