// hybrid_sig.go
// hybrid signature cryptography for the go std library using liboqs-go.

package hybrid

import (
	"crypto/ecdsa"
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
)

const (
	HYBRID_SIGNATURE uint8 = iota
	PQC_SIGNATURE
	CLASSIC_SIGNATURE
)

type SigName struct {
	pqc     string
	classic elliptic.Curve
}

var SigOIDtoName = map[OID]SigName{
	MAYO1_P256: {"Mayo-1", elliptic.P256()},
	MAYO2_P256: {"Mayo-2", elliptic.P256()},
	MAYO3_P384: {"Mayo-3", elliptic.P384()},
	MAYO5_P521: {"Mayo-5", elliptic.P521()},
}

type PublicKey struct {
	SigOID  OID
	pqc     []byte
	classic *ecdsa.PublicKey
}

type PrivateKey struct {
	SigOID  OID
	pqc     []byte
	classic *ecdsa.PrivateKey
	public  *PublicKey
}

func (pub *PublicKey) GetPublicKeys() (pqc []byte, classic *ecdsa.PublicKey) {
	return pub.pqc, pub.classic
}

func (priv *PrivateKey) GetPrivateKeys() (pqc []byte, classic *ecdsa.PrivateKey) {
	return priv.pqc, priv.classic
}

func (priv *PrivateKey) ExportPublicKey() (pub PublicKey) {
	return *priv.public
}

func GenerateKey(sigOID OID) (priv PrivateKey, err error) {
	var pub PublicKey

	// Post-quantum LibOQS
	pqcSigner := oqs.Signature{}
	if err = pqcSigner.Init(SigOIDtoName[sigOID].pqc, nil); err != nil {
		return PrivateKey{}, err
	}
	if pub.pqc, err = pqcSigner.GenerateKeyPair(); err != nil {
		return PrivateKey{}, err
	}
	priv.pqc = pqcSigner.ExportSecretKey()

	// Classic NIST curves
	if priv.classic, err = ecdsa.GenerateKey(SigOIDtoName[sigOID].classic, rand.Reader); err != nil {
		return PrivateKey{}, err
	}
	pub.classic = &priv.classic.PublicKey

	pub.SigOID = sigOID
	priv.SigOID = sigOID
	priv.public = &pub

	return priv, err
}

func (priv *PrivateKey) Sign(message []byte) (signature []byte, err error) {
	var hybridSig cryptobyte.Builder
	var pqcSig []byte
	var classicSig []byte

	// Post-quantum LibOQS
	pqcSigner := oqs.Signature{}
	if err := pqcSigner.Init(SigOIDtoName[priv.SigOID].pqc, priv.pqc); err != nil {
		return nil, err
	}
	if pqcSig, err = pqcSigner.Sign(message); err != nil {
		return nil, err
	}

	// Classic NIST curves
	if classicSig, err = ecdsa.SignASN1(rand.Reader, priv.classic, message); err != nil {
		return nil, err
	}

	// Concat both to make the hybrid signature
	hybridSig.AddUint16(uint16(len(pqcSig)))
	hybridSig.AddBytes(pqcSig)
	hybridSig.AddUint16(uint16(len(classicSig)))
	hybridSig.AddBytes(classicSig)

	return hybridSig.BytesOrPanic(), err
}

func VerifyClassic(pub PublicKey, message []byte, signature []byte) bool {
	return ecdsa.VerifyASN1(pub.classic, message, signature)
}

func VerifyPQC(pub PublicKey, message []byte, signature []byte) bool {
	var valid bool
	var err error

	var verifier oqs.Signature
	if err = verifier.Init(SigOIDtoName[pub.SigOID].pqc, nil); err != nil {
		return false
	}

	if valid, err = verifier.Verify(message, signature, pub.pqc); err != nil {
		return false
	}

	return valid
}

func Verify(option uint8, pub PublicKey, message []byte, signature []byte) bool {
	var current uint16 = 0

	pqcSize := binary.BigEndian.Uint16(signature[current : current+2])
	current = current + 2
	pqcSig := signature[current : current+pqcSize]
	current = current + pqcSize

	if option == PQC_SIGNATURE {
		return VerifyPQC(pub, message, pqcSig)
	}

	classicSize := binary.BigEndian.Uint16(signature[current : current+2])
	current = current + 2
	classicSig := signature[current : current+classicSize]
	current = current + classicSize

	switch option {
	case CLASSIC_SIGNATURE:
		return VerifyClassic(pub, message, classicSig)

	default:
		return VerifyPQC(pub, message, pqcSig) && VerifyClassic(pub, message, classicSig)
	}

	return false
}
