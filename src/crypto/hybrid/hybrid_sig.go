// hybrid_sig.go
// hybrid signature cryptography for the go std library using liboqs-go.

package hybrid

import (
	"crypto/ecdsa"
	"crypto/ed25519"
	"crypto/elliptic"
	"crypto/rand"

	"encoding/binary"

	"github.com/open-quantum-safe/liboqs-go/oqs"
	"golang.org/x/crypto/cryptobyte"
)

// Object IDentifiers for PQC and Composite
// https://github.com/IETF-Hackathon/pqc-certificates/blob/master/docs/oid_mapping.md
type OID string

const (
	MAYO1_P256 OID = "0.1.1.1" // TODO: use proper OID values
	MAYO2_P256 OID = "0.1.2.2"
	MAYO3_P384 OID = "0.1.3.3"
	MAYO5_P521 OID = "0.1.5.4"

	MAYO1_ED25519 OID = "0.1.1.5"
	MAYO2_ED25519 OID = "0.1.2.5"
	MAYO3_ED25519 OID = "0.1.3.5"
	MAYO5_ED25519 OID = "0.1.5.5"

	CROSS_128_SMALL_P256 OID = "0.2.1.1.S"
	CROSS_128_FAST_P256  OID = "0.2.1.2.F"
	CROSS_192_SMALL_P384 OID = "0.2.3.3.S"
	CROSS_256_SMALL_P521 OID = "0.2.5.4.S"

	CROSS_128_SMALL_ED25519 OID = "0.2.1.5.S"
	CROSS_128_FAST_ED25519  OID = "0.2.1.5.F"
	CROSS_192_SMALL_ED25519 OID = "0.2.3.5.S"
	CROSS_256_SMALL_ED25519 OID = "0.2.5.5.S"

	ML_DSA_44_P256 OID = "0.3.1.2"
	ML_DSA_65_P384 OID = "0.3.1.3"
	ML_DSA_87_P521 OID = "0.3.1.5"

	ML_DSA_65_ED25519 OID = "0.3.2.2"
)

type SigName struct {
	pqc     string
	classic interface{}
}

var SigOIDtoName = map[OID]SigName{
	MAYO1_P256: {"Mayo-1", elliptic.P256()},
	MAYO2_P256: {"Mayo-2", elliptic.P256()},
	MAYO3_P384: {"Mayo-3", elliptic.P384()},
	MAYO5_P521: {"Mayo-5", elliptic.P521()},

	MAYO1_ED25519: {"Mayo-1", ed25519.PrivateKey{}},
	MAYO2_ED25519: {"Mayo-2", ed25519.PrivateKey{}},
	MAYO3_ED25519: {"Mayo-3", ed25519.PrivateKey{}},
	MAYO5_ED25519: {"Mayo-5", ed25519.PrivateKey{}},

	CROSS_128_SMALL_P256: {"cross-rsdpg-128-small", elliptic.P256()},
	CROSS_128_FAST_P256:  {"cross-rsdpg-128-fast", elliptic.P256()},
	CROSS_192_SMALL_P384: {"cross-rsdpg-192-small", elliptic.P384()},
	CROSS_256_SMALL_P521: {"cross-rsdpg-256-small", elliptic.P521()},

	CROSS_128_SMALL_ED25519: {"cross-rsdpg-128-small", ed25519.PrivateKey{}},
	CROSS_128_FAST_ED25519:  {"cross-rsdpg-128-fast", ed25519.PrivateKey{}},
	CROSS_192_SMALL_ED25519: {"cross-rsdpg-192-small", ed25519.PrivateKey{}},
	CROSS_256_SMALL_ED25519: {"cross-rsdpg-256-small", ed25519.PrivateKey{}},

	ML_DSA_44_P256: {"ML-DSA-44", elliptic.P256()},
	ML_DSA_65_P384: {"ML-DSA-65", elliptic.P384()},
	ML_DSA_87_P521: {"ML-DSA-87", elliptic.P521()},

	ML_DSA_65_ED25519: {"ML-DSA-65", ed25519.PrivateKey{}},
}

type PublicKey struct {
	SigOID  OID
	classic *interface{}
	pqc     []byte
}

type PrivateKey struct {
	SigOID  OID
	classic *interface{}
	pqc     []byte
	public  *PublicKey
}

// GetPublicKeys returns the classic and post-quantum private keys from a sig receiver.
func (pub *PublicKey) GetPublicKeys() (classic *interface{}, pqc []byte) {
	return pub.classic, pub.pqc
}

// GetPrivateKeys returns the classic and post-quantum private keys from a sig receiver.
func (priv *PrivateKey) GetPrivateKeys() (classic *interface{}, pqc []byte) {
	return priv.classic, priv.pqc
}

// ExportPublicKey exports the corresponding public hybrid key from the sig receiver.
func (priv *PrivateKey) ExportPublicKey() (pub PublicKey) {
	return *priv.public
}

// GenerateKey generates a pair of private and public hybrid keys, and returns the private key.
// The public key is stored inside the sig receiver. The public key is not directly acessible, unless one
// exports it with the Signature.ExportPublicKey method.
func GenerateKey(sigOID OID) (priv PrivateKey, err error) {
	var pub PublicKey

	pub.classic = new(interface{})
	priv.classic = new(interface{})

	switch classic := SigOIDtoName[sigOID].classic.(type) {
	case elliptic.Curve:
		if (*priv.classic), err = ecdsa.GenerateKey(classic, rand.Reader); err != nil {
			return PrivateKey{}, err
		}
		*pub.classic = &((*priv.classic).(*ecdsa.PrivateKey)).PublicKey

	case ed25519.PrivateKey:
		if (*pub.classic), (*priv.classic), err = ed25519.GenerateKey(rand.Reader); err != nil {
			return PrivateKey{}, err
		}
	}

	pqcSigner := oqs.Signature{}
	if err = pqcSigner.Init(SigOIDtoName[sigOID].pqc, nil); err != nil {
		return PrivateKey{}, err
	}
	if pub.pqc, err = pqcSigner.GenerateKeyPair(); err != nil {
		return PrivateKey{}, err
	}
	priv.pqc = pqcSigner.ExportSecretKey()

	pub.SigOID = sigOID
	priv.SigOID = sigOID
	priv.public = &pub

	return priv, err
}

// Sign signs a hash (which should be the result of hashing a larger message) using the private
// key, and returns the corresponding signature.
func (priv *PrivateKey) Sign(hash []byte) (signature []byte, err error) {
	var hybridSig cryptobyte.Builder
	var pqcSig []byte
	var classicSig []byte

	switch classic := (*priv.classic).(type) {
	case elliptic.Curve:
		if classicSig, err = ecdsa.SignASN1(rand.Reader, (*priv.classic).(*ecdsa.PrivateKey), hash); err != nil {
			return nil, err
		}
	case ed25519.PrivateKey:
		if classicSig = ed25519.Sign(classic, hash); err != nil {
			return nil, err
		}
	}

	pqcSigner := oqs.Signature{}
	if err := pqcSigner.Init(SigOIDtoName[priv.SigOID].pqc, priv.pqc); err != nil {
		return nil, err
	}
	if pqcSig, err = pqcSigner.Sign(hash); err != nil {
		return nil, err
	}

	hybridSig.AddUint16(uint16(len(classicSig)))
	hybridSig.AddBytes(classicSig)
	hybridSig.AddUint16(uint16(len(pqcSig)))
	hybridSig.AddBytes(pqcSig)

	return hybridSig.BytesOrPanic(), err
}

func verifyClassic(pub PublicKey, hash []byte, signature []byte) bool {
	switch classic := (*pub.classic).(type) {
	case elliptic.Curve:
		return ecdsa.VerifyASN1((*pub.classic).(*ecdsa.PublicKey), hash, signature)
	case ed25519.PublicKey:
		return ed25519.Verify((*pub.classic).(ed25519.PublicKey), hash, signature)
		_ = classic // UNREACHABLE
	}
	return false
}

func verifyPQC(pub PublicKey, hash []byte, signature []byte) bool {
	var valid bool
	var err error

	var verifier oqs.Signature
	if err = verifier.Init(SigOIDtoName[pub.SigOID].pqc, nil); err != nil {
		return false
	}

	if valid, err = verifier.Verify(hash, signature, pub.pqc); err != nil {
		return false
	}

	return valid
}

// VerifyHybrid verifies the validity of a signed hash, returning true if the hybrid signature is valid, and false otherwise.
func VerifyHybrid(pub PublicKey, hash []byte, signature []byte) bool {
	var current uint16 = 0

	classicSize := binary.BigEndian.Uint16(signature[current : current+2])
	current = current + 2
	classicSig := signature[current : current+classicSize]
	current = current + classicSize

	pqcSize := binary.BigEndian.Uint16(signature[current : current+2])
	current = current + 2
	pqcSig := signature[current : current+pqcSize]
	current = current + pqcSize

	return verifyClassic(pub, hash, classicSig) && verifyPQC(pub, hash, pqcSig)
}
