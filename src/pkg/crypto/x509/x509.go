// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package x509 parses X.509-encoded keys and certificates.
package x509

import (
	"asn1"
	"big"
	"bytes"
	"crypto"
	"crypto/dsa"
	"crypto/rsa"
	"crypto/sha1"
	"crypto/x509/pkix"
	"encoding/pem"
	"errors"
	"io"
	"time"
)

// pkixPublicKey reflects a PKIX public key structure. See SubjectPublicKeyInfo
// in RFC 3280.
type pkixPublicKey struct {
	Algo      pkix.AlgorithmIdentifier
	BitString asn1.BitString
}

// ParsePKIXPublicKey parses a DER encoded public key. These values are
// typically found in PEM blocks with "BEGIN PUBLIC KEY".
func ParsePKIXPublicKey(derBytes []byte) (pub interface{}, err error) {
	var pki publicKeyInfo
	if _, err = asn1.Unmarshal(derBytes, &pki); err != nil {
		return
	}
	algo := getPublicKeyAlgorithmFromOID(pki.Algorithm.Algorithm)
	if algo == UnknownPublicKeyAlgorithm {
		return nil, errors.New("ParsePKIXPublicKey: unknown public key algorithm")
	}
	return parsePublicKey(algo, &pki)
}

// MarshalPKIXPublicKey serialises a public key to DER-encoded PKIX format.
func MarshalPKIXPublicKey(pub interface{}) ([]byte, error) {
	var pubBytes []byte

	switch pub := pub.(type) {
	case *rsa.PublicKey:
		pubBytes, _ = asn1.Marshal(rsaPublicKey{
			N: pub.N,
			E: pub.E,
		})
	default:
		return nil, errors.New("MarshalPKIXPublicKey: unknown public key type")
	}

	pkix := pkixPublicKey{
		Algo: pkix.AlgorithmIdentifier{
			Algorithm: []int{1, 2, 840, 113549, 1, 1, 1},
			// This is a NULL parameters value which is technically
			// superfluous, but most other code includes it and, by
			// doing this, we match their public key hashes.
			Parameters: asn1.RawValue{
				Tag: 5,
			},
		},
		BitString: asn1.BitString{
			Bytes:     pubBytes,
			BitLength: 8 * len(pubBytes),
		},
	}

	ret, _ := asn1.Marshal(pkix)
	return ret, nil
}

// These structures reflect the ASN.1 structure of X.509 certificates.:

type certificate struct {
	Raw                asn1.RawContent
	TBSCertificate     tbsCertificate
	SignatureAlgorithm pkix.AlgorithmIdentifier
	SignatureValue     asn1.BitString
}

type tbsCertificate struct {
	Raw                asn1.RawContent
	Version            int `asn1:"optional,explicit,default:1,tag:0"`
	SerialNumber       *big.Int
	SignatureAlgorithm pkix.AlgorithmIdentifier
	Issuer             asn1.RawValue
	Validity           validity
	Subject            asn1.RawValue
	PublicKey          publicKeyInfo
	UniqueId           asn1.BitString   `asn1:"optional,tag:1"`
	SubjectUniqueId    asn1.BitString   `asn1:"optional,tag:2"`
	Extensions         []pkix.Extension `asn1:"optional,explicit,tag:3"`
}

type dsaAlgorithmParameters struct {
	P, Q, G *big.Int
}

type dsaSignature struct {
	R, S *big.Int
}

type validity struct {
	NotBefore, NotAfter *time.Time
}

type publicKeyInfo struct {
	Raw       asn1.RawContent
	Algorithm pkix.AlgorithmIdentifier
	PublicKey asn1.BitString
}

// RFC 5280,  4.2.1.1
type authKeyId struct {
	Id []byte `asn1:"optional,tag:0"`
}

type SignatureAlgorithm int

const (
	UnknownSignatureAlgorithm SignatureAlgorithm = iota
	MD2WithRSA
	MD5WithRSA
	SHA1WithRSA
	SHA256WithRSA
	SHA384WithRSA
	SHA512WithRSA
	DSAWithSHA1
	DSAWithSHA256
)

type PublicKeyAlgorithm int

const (
	UnknownPublicKeyAlgorithm PublicKeyAlgorithm = iota
	RSA
	DSA
)

// OIDs for signature algorithms
//
// pkcs-1 OBJECT IDENTIFIER ::= {
//    iso(1) member-body(2) us(840) rsadsi(113549) pkcs(1) 1 }
// 
// 
// RFC 3279 2.2.1 RSA Signature Algorithms
//
// md2WithRSAEncryption OBJECT IDENTIFIER ::= { pkcs-1 2 }
//
// md5WithRSAEncryption OBJECT IDENTIFER ::= { pkcs-1 4 }
//
// sha-1WithRSAEncryption OBJECT IDENTIFIER ::= { pkcs-1 5 }
// 
// dsaWithSha1 OBJECT IDENTIFIER ::= {
//    iso(1) member-body(2) us(840) x9-57(10040) x9cm(4) 3 } 
//
//
// RFC 4055 5 PKCS #1 Version 1.5
// 
// sha256WithRSAEncryption OBJECT IDENTIFIER ::= { pkcs-1 11 }
//
// sha384WithRSAEncryption OBJECT IDENTIFIER ::= { pkcs-1 12 }
//
// sha512WithRSAEncryption OBJECT IDENTIFIER ::= { pkcs-1 13 }
//
//
// RFC 5758 3.1 DSA Signature Algorithms
//
// dsaWithSha356 OBJECT IDENTIFER ::= {
//    joint-iso-ccitt(2) country(16) us(840) organization(1) gov(101)
//    algorithms(4) id-dsa-with-sha2(3) 2}
//
var (
	oidSignatureMD2WithRSA    = asn1.ObjectIdentifier{1, 2, 840, 113549, 1, 1, 2}
	oidSignatureMD5WithRSA    = asn1.ObjectIdentifier{1, 2, 840, 113549, 1, 1, 4}
	oidSignatureSHA1WithRSA   = asn1.ObjectIdentifier{1, 2, 840, 113549, 1, 1, 5}
	oidSignatureSHA256WithRSA = asn1.ObjectIdentifier{1, 2, 840, 113549, 1, 1, 11}
	oidSignatureSHA384WithRSA = asn1.ObjectIdentifier{1, 2, 840, 113549, 1, 1, 12}
	oidSignatureSHA512WithRSA = asn1.ObjectIdentifier{1, 2, 840, 113549, 1, 1, 13}
	oidSignatureDSAWithSHA1   = asn1.ObjectIdentifier{1, 2, 840, 10040, 4, 3}
	oidSignatureDSAWithSHA256 = asn1.ObjectIdentifier{2, 16, 840, 1, 101, 4, 3, 2}
)

func getSignatureAlgorithmFromOID(oid asn1.ObjectIdentifier) SignatureAlgorithm {
	switch {
	case oid.Equal(oidSignatureMD2WithRSA):
		return MD2WithRSA
	case oid.Equal(oidSignatureMD5WithRSA):
		return MD5WithRSA
	case oid.Equal(oidSignatureSHA1WithRSA):
		return SHA1WithRSA
	case oid.Equal(oidSignatureSHA256WithRSA):
		return SHA256WithRSA
	case oid.Equal(oidSignatureSHA384WithRSA):
		return SHA384WithRSA
	case oid.Equal(oidSignatureSHA512WithRSA):
		return SHA512WithRSA
	case oid.Equal(oidSignatureDSAWithSHA1):
		return DSAWithSHA1
	case oid.Equal(oidSignatureDSAWithSHA256):
		return DSAWithSHA256
	}
	return UnknownSignatureAlgorithm
}

// RFC 3279, 2.3 Public Key Algorithms
//
// pkcs-1 OBJECT IDENTIFIER ::== { iso(1) member-body(2) us(840)
//    rsadsi(113549) pkcs(1) 1 }
//
// rsaEncryption OBJECT IDENTIFIER ::== { pkcs1-1 1 }
//
// id-dsa OBJECT IDENTIFIER ::== { iso(1) member-body(2) us(840)
//    x9-57(10040) x9cm(4) 1 }
var (
	oidPublicKeyRsa = asn1.ObjectIdentifier{1, 2, 840, 113549, 1, 1, 1}
	oidPublicKeyDsa = asn1.ObjectIdentifier{1, 2, 840, 10040, 4, 1}
)

func getPublicKeyAlgorithmFromOID(oid asn1.ObjectIdentifier) PublicKeyAlgorithm {
	switch {
	case oid.Equal(oidPublicKeyRsa):
		return RSA
	case oid.Equal(oidPublicKeyDsa):
		return DSA
	}
	return UnknownPublicKeyAlgorithm
}

// KeyUsage represents the set of actions that are valid for a given key. It's
// a bitmap of the KeyUsage* constants.
type KeyUsage int

const (
	KeyUsageDigitalSignature KeyUsage = 1 << iota
	KeyUsageContentCommitment
	KeyUsageKeyEncipherment
	KeyUsageDataEncipherment
	KeyUsageKeyAgreement
	KeyUsageCertSign
	KeyUsageCRLSign
	KeyUsageEncipherOnly
	KeyUsageDecipherOnly
)

// RFC 5280, 4.2.1.12  Extended Key Usage
//
// anyExtendedKeyUsage OBJECT IDENTIFIER ::= { id-ce-extKeyUsage 0 }
//
// id-kp OBJECT IDENTIFIER ::= { id-pkix 3 }
//
// id-kp-serverAuth             OBJECT IDENTIFIER ::= { id-kp 1 }
// id-kp-clientAuth             OBJECT IDENTIFIER ::= { id-kp 2 }
// id-kp-codeSigning            OBJECT IDENTIFIER ::= { id-kp 3 }
// id-kp-emailProtection        OBJECT IDENTIFIER ::= { id-kp 4 }
// id-kp-timeStamping           OBJECT IDENTIFIER ::= { id-kp 8 }
// id-kp-OCSPSigning            OBJECT IDENTIFIER ::= { id-kp 9 }
var (
	oidExtKeyUsageAny             = asn1.ObjectIdentifier{2, 5, 29, 37, 0}
	oidExtKeyUsageServerAuth      = asn1.ObjectIdentifier{1, 3, 6, 1, 5, 5, 7, 3, 1}
	oidExtKeyUsageClientAuth      = asn1.ObjectIdentifier{1, 3, 6, 1, 5, 5, 7, 3, 2}
	oidExtKeyUsageCodeSigning     = asn1.ObjectIdentifier{1, 3, 6, 1, 5, 5, 7, 3, 3}
	oidExtKeyUsageEmailProtection = asn1.ObjectIdentifier{1, 3, 6, 1, 5, 5, 7, 3, 4}
	oidExtKeyUsageTimeStamping    = asn1.ObjectIdentifier{1, 3, 6, 1, 5, 5, 7, 3, 8}
	oidExtKeyUsageOCSPSigning     = asn1.ObjectIdentifier{1, 3, 6, 1, 5, 5, 7, 3, 9}
)

// ExtKeyUsage represents an extended set of actions that are valid for a given key.
// Each of the ExtKeyUsage* constants define a unique action.
type ExtKeyUsage int

const (
	ExtKeyUsageAny ExtKeyUsage = iota
	ExtKeyUsageServerAuth
	ExtKeyUsageClientAuth
	ExtKeyUsageCodeSigning
	ExtKeyUsageEmailProtection
	ExtKeyUsageTimeStamping
	ExtKeyUsageOCSPSigning
)

// A Certificate represents an X.509 certificate.
type Certificate struct {
	Raw                     []byte // Complete ASN.1 DER content (certificate, signature algorithm and signature).
	RawTBSCertificate       []byte // Certificate part of raw ASN.1 DER content.
	RawSubjectPublicKeyInfo []byte // DER encoded SubjectPublicKeyInfo.
	RawSubject              []byte // DER encoded Subject
	RawIssuer               []byte // DER encoded Issuer

	Signature          []byte
	SignatureAlgorithm SignatureAlgorithm

	PublicKeyAlgorithm PublicKeyAlgorithm
	PublicKey          interface{}

	Version             int
	SerialNumber        *big.Int
	Issuer              pkix.Name
	Subject             pkix.Name
	NotBefore, NotAfter *time.Time // Validity bounds.
	KeyUsage            KeyUsage

	ExtKeyUsage        []ExtKeyUsage           // Sequence of extended key usages.
	UnknownExtKeyUsage []asn1.ObjectIdentifier // Encountered extended key usages unknown to this package.

	BasicConstraintsValid bool // if true then the next two fields are valid.
	IsCA                  bool
	MaxPathLen            int

	SubjectKeyId   []byte
	AuthorityKeyId []byte

	// Subject Alternate Name values
	DNSNames       []string
	EmailAddresses []string

	// Name constraints
	PermittedDNSDomainsCritical bool // if true then the name constraints are marked critical.
	PermittedDNSDomains         []string

	PolicyIdentifiers []asn1.ObjectIdentifier
}

// UnsupportedAlgorithmError results from attempting to perform an operation
// that involves algorithms that are not currently implemented.
type UnsupportedAlgorithmError struct{}

func (UnsupportedAlgorithmError) Error() string {
	return "cannot verify signature: algorithm unimplemented"
}

// ConstraintViolationError results when a requested usage is not permitted by
// a certificate. For example: checking a signature when the public key isn't a
// certificate signing key.
type ConstraintViolationError struct{}

func (ConstraintViolationError) Error() string {
	return "invalid signature: parent certificate cannot sign this kind of certificate"
}

func (c *Certificate) Equal(other *Certificate) bool {
	return bytes.Equal(c.Raw, other.Raw)
}

// CheckSignatureFrom verifies that the signature on c is a valid signature
// from parent.
func (c *Certificate) CheckSignatureFrom(parent *Certificate) (err error) {
	// RFC 5280, 4.2.1.9:
	// "If the basic constraints extension is not present in a version 3
	// certificate, or the extension is present but the cA boolean is not
	// asserted, then the certified public key MUST NOT be used to verify
	// certificate signatures."
	if parent.Version == 3 && !parent.BasicConstraintsValid ||
		parent.BasicConstraintsValid && !parent.IsCA {
		return ConstraintViolationError{}
	}

	if parent.KeyUsage != 0 && parent.KeyUsage&KeyUsageCertSign == 0 {
		return ConstraintViolationError{}
	}

	if parent.PublicKeyAlgorithm == UnknownPublicKeyAlgorithm {
		return UnsupportedAlgorithmError{}
	}

	// TODO(agl): don't ignore the path length constraint.

	return parent.CheckSignature(c.SignatureAlgorithm, c.RawTBSCertificate, c.Signature)
}

// CheckSignature verifies that signature is a valid signature over signed from
// c's public key.
func (c *Certificate) CheckSignature(algo SignatureAlgorithm, signed, signature []byte) (err error) {
	var hashType crypto.Hash

	switch algo {
	case SHA1WithRSA, DSAWithSHA1:
		hashType = crypto.SHA1
	case SHA256WithRSA, DSAWithSHA256:
		hashType = crypto.SHA256
	case SHA384WithRSA:
		hashType = crypto.SHA384
	case SHA512WithRSA:
		hashType = crypto.SHA512
	default:
		return UnsupportedAlgorithmError{}
	}

	h := hashType.New()
	if h == nil {
		return UnsupportedAlgorithmError{}
	}

	h.Write(signed)
	digest := h.Sum()

	switch pub := c.PublicKey.(type) {
	case *rsa.PublicKey:
		return rsa.VerifyPKCS1v15(pub, hashType, digest, signature)
	case *dsa.PublicKey:
		dsaSig := new(dsaSignature)
		if _, err := asn1.Unmarshal(signature, dsaSig); err != nil {
			return err
		}
		if dsaSig.R.Sign() <= 0 || dsaSig.S.Sign() <= 0 {
			return errors.New("DSA signature contained zero or negative values")
		}
		if !dsa.Verify(pub, digest, dsaSig.R, dsaSig.S) {
			return errors.New("DSA verification failure")
		}
		return
	}
	return UnsupportedAlgorithmError{}
}

// CheckCRLSignature checks that the signature in crl is from c.
func (c *Certificate) CheckCRLSignature(crl *pkix.CertificateList) (err error) {
	algo := getSignatureAlgorithmFromOID(crl.SignatureAlgorithm.Algorithm)
	return c.CheckSignature(algo, crl.TBSCertList.Raw, crl.SignatureValue.RightAlign())
}

type UnhandledCriticalExtension struct{}

func (h UnhandledCriticalExtension) Error() string {
	return "unhandled critical extension"
}

type basicConstraints struct {
	IsCA       bool `asn1:"optional"`
	MaxPathLen int  `asn1:"optional"`
}

// RFC 5280 4.2.1.4
type policyInformation struct {
	Policy asn1.ObjectIdentifier
	// policyQualifiers omitted
}

// RFC 5280, 4.2.1.10
type nameConstraints struct {
	Permitted []generalSubtree `asn1:"optional,tag:0"`
	Excluded  []generalSubtree `asn1:"optional,tag:1"`
}

type generalSubtree struct {
	Name string `asn1:"tag:2,optional,ia5"`
	Min  int    `asn1:"optional,tag:0"`
	Max  int    `asn1:"optional,tag:1"`
}

func parsePublicKey(algo PublicKeyAlgorithm, keyData *publicKeyInfo) (interface{}, error) {
	asn1Data := keyData.PublicKey.RightAlign()
	switch algo {
	case RSA:
		p := new(rsaPublicKey)
		_, err := asn1.Unmarshal(asn1Data, p)
		if err != nil {
			return nil, err
		}

		pub := &rsa.PublicKey{
			E: p.E,
			N: p.N,
		}
		return pub, nil
	case DSA:
		var p *big.Int
		_, err := asn1.Unmarshal(asn1Data, &p)
		if err != nil {
			return nil, err
		}
		paramsData := keyData.Algorithm.Parameters.FullBytes
		params := new(dsaAlgorithmParameters)
		_, err = asn1.Unmarshal(paramsData, params)
		if err != nil {
			return nil, err
		}
		if p.Sign() <= 0 || params.P.Sign() <= 0 || params.Q.Sign() <= 0 || params.G.Sign() <= 0 {
			return nil, errors.New("zero or negative DSA parameter")
		}
		pub := &dsa.PublicKey{
			Parameters: dsa.Parameters{
				P: params.P,
				Q: params.Q,
				G: params.G,
			},
			Y: p,
		}
		return pub, nil
	default:
		return nil, nil
	}
	panic("unreachable")
}

func parseCertificate(in *certificate) (*Certificate, error) {
	out := new(Certificate)
	out.Raw = in.Raw
	out.RawTBSCertificate = in.TBSCertificate.Raw
	out.RawSubjectPublicKeyInfo = in.TBSCertificate.PublicKey.Raw
	out.RawSubject = in.TBSCertificate.Subject.FullBytes
	out.RawIssuer = in.TBSCertificate.Issuer.FullBytes

	out.Signature = in.SignatureValue.RightAlign()
	out.SignatureAlgorithm =
		getSignatureAlgorithmFromOID(in.TBSCertificate.SignatureAlgorithm.Algorithm)

	out.PublicKeyAlgorithm =
		getPublicKeyAlgorithmFromOID(in.TBSCertificate.PublicKey.Algorithm.Algorithm)
	var err error
	out.PublicKey, err = parsePublicKey(out.PublicKeyAlgorithm, &in.TBSCertificate.PublicKey)
	if err != nil {
		return nil, err
	}

	if in.TBSCertificate.SerialNumber.Sign() < 0 {
		return nil, errors.New("negative serial number")
	}

	out.Version = in.TBSCertificate.Version + 1
	out.SerialNumber = in.TBSCertificate.SerialNumber

	var issuer, subject pkix.RDNSequence
	if _, err := asn1.Unmarshal(in.TBSCertificate.Subject.FullBytes, &subject); err != nil {
		return nil, err
	}
	if _, err := asn1.Unmarshal(in.TBSCertificate.Issuer.FullBytes, &issuer); err != nil {
		return nil, err
	}

	out.Issuer.FillFromRDNSequence(&issuer)
	out.Subject.FillFromRDNSequence(&subject)

	out.NotBefore = in.TBSCertificate.Validity.NotBefore
	out.NotAfter = in.TBSCertificate.Validity.NotAfter

	for _, e := range in.TBSCertificate.Extensions {
		if len(e.Id) == 4 && e.Id[0] == 2 && e.Id[1] == 5 && e.Id[2] == 29 {
			switch e.Id[3] {
			case 15:
				// RFC 5280, 4.2.1.3
				var usageBits asn1.BitString
				_, err := asn1.Unmarshal(e.Value, &usageBits)

				if err == nil {
					var usage int
					for i := 0; i < 9; i++ {
						if usageBits.At(i) != 0 {
							usage |= 1 << uint(i)
						}
					}
					out.KeyUsage = KeyUsage(usage)
					continue
				}
			case 19:
				// RFC 5280, 4.2.1.9
				var constraints basicConstraints
				_, err := asn1.Unmarshal(e.Value, &constraints)

				if err == nil {
					out.BasicConstraintsValid = true
					out.IsCA = constraints.IsCA
					out.MaxPathLen = constraints.MaxPathLen
					continue
				}
			case 17:
				// RFC 5280, 4.2.1.6

				// SubjectAltName ::= GeneralNames
				//
				// GeneralNames ::= SEQUENCE SIZE (1..MAX) OF GeneralName
				//
				// GeneralName ::= CHOICE {
				//      otherName                       [0]     OtherName,
				//      rfc822Name                      [1]     IA5String,
				//      dNSName                         [2]     IA5String,
				//      x400Address                     [3]     ORAddress,
				//      directoryName                   [4]     Name,
				//      ediPartyName                    [5]     EDIPartyName,
				//      uniformResourceIdentifier       [6]     IA5String,
				//      iPAddress                       [7]     OCTET STRING,
				//      registeredID                    [8]     OBJECT IDENTIFIER }
				var seq asn1.RawValue
				_, err := asn1.Unmarshal(e.Value, &seq)
				if err != nil {
					return nil, err
				}
				if !seq.IsCompound || seq.Tag != 16 || seq.Class != 0 {
					return nil, asn1.StructuralError{"bad SAN sequence"}
				}

				parsedName := false

				rest := seq.Bytes
				for len(rest) > 0 {
					var v asn1.RawValue
					rest, err = asn1.Unmarshal(rest, &v)
					if err != nil {
						return nil, err
					}
					switch v.Tag {
					case 1:
						out.EmailAddresses = append(out.EmailAddresses, string(v.Bytes))
						parsedName = true
					case 2:
						out.DNSNames = append(out.DNSNames, string(v.Bytes))
						parsedName = true
					}
				}

				if parsedName {
					continue
				}
				// If we didn't parse any of the names then we
				// fall through to the critical check below.

			case 30:
				// RFC 5280, 4.2.1.10

				// NameConstraints ::= SEQUENCE {
				//      permittedSubtrees       [0]     GeneralSubtrees OPTIONAL,
				//      excludedSubtrees        [1]     GeneralSubtrees OPTIONAL }
				//
				// GeneralSubtrees ::= SEQUENCE SIZE (1..MAX) OF GeneralSubtree
				//
				// GeneralSubtree ::= SEQUENCE {
				//      base                    GeneralName,
				//      minimum         [0]     BaseDistance DEFAULT 0,
				//      maximum         [1]     BaseDistance OPTIONAL }
				//
				// BaseDistance ::= INTEGER (0..MAX)

				var constraints nameConstraints
				_, err := asn1.Unmarshal(e.Value, &constraints)
				if err != nil {
					return nil, err
				}

				if len(constraints.Excluded) > 0 && e.Critical {
					return out, UnhandledCriticalExtension{}
				}

				for _, subtree := range constraints.Permitted {
					if subtree.Min > 0 || subtree.Max > 0 || len(subtree.Name) == 0 {
						if e.Critical {
							return out, UnhandledCriticalExtension{}
						}
						continue
					}
					out.PermittedDNSDomains = append(out.PermittedDNSDomains, subtree.Name)
				}
				continue

			case 35:
				// RFC 5280, 4.2.1.1
				var a authKeyId
				_, err = asn1.Unmarshal(e.Value, &a)
				if err != nil {
					return nil, err
				}
				out.AuthorityKeyId = a.Id
				continue

			case 37:
				// RFC 5280, 4.2.1.12.  Extended Key Usage

				// id-ce-extKeyUsage OBJECT IDENTIFIER ::= { id-ce 37 }
				//
				// ExtKeyUsageSyntax ::= SEQUENCE SIZE (1..MAX) OF KeyPurposeId
				//
				// KeyPurposeId ::= OBJECT IDENTIFIER

				var keyUsage []asn1.ObjectIdentifier
				_, err = asn1.Unmarshal(e.Value, &keyUsage)
				if err != nil {
					return nil, err
				}

				for _, u := range keyUsage {
					switch {
					case u.Equal(oidExtKeyUsageAny):
						out.ExtKeyUsage = append(out.ExtKeyUsage, ExtKeyUsageAny)
					case u.Equal(oidExtKeyUsageServerAuth):
						out.ExtKeyUsage = append(out.ExtKeyUsage, ExtKeyUsageServerAuth)
					case u.Equal(oidExtKeyUsageClientAuth):
						out.ExtKeyUsage = append(out.ExtKeyUsage, ExtKeyUsageClientAuth)
					case u.Equal(oidExtKeyUsageCodeSigning):
						out.ExtKeyUsage = append(out.ExtKeyUsage, ExtKeyUsageCodeSigning)
					case u.Equal(oidExtKeyUsageEmailProtection):
						out.ExtKeyUsage = append(out.ExtKeyUsage, ExtKeyUsageEmailProtection)
					case u.Equal(oidExtKeyUsageTimeStamping):
						out.ExtKeyUsage = append(out.ExtKeyUsage, ExtKeyUsageTimeStamping)
					case u.Equal(oidExtKeyUsageOCSPSigning):
						out.ExtKeyUsage = append(out.ExtKeyUsage, ExtKeyUsageOCSPSigning)
					default:
						out.UnknownExtKeyUsage = append(out.UnknownExtKeyUsage, u)
					}
				}

				continue

			case 14:
				// RFC 5280, 4.2.1.2
				var keyid []byte
				_, err = asn1.Unmarshal(e.Value, &keyid)
				if err != nil {
					return nil, err
				}
				out.SubjectKeyId = keyid
				continue

			case 32:
				// RFC 5280 4.2.1.4: Certificate Policies
				var policies []policyInformation
				if _, err = asn1.Unmarshal(e.Value, &policies); err != nil {
					return nil, err
				}
				out.PolicyIdentifiers = make([]asn1.ObjectIdentifier, len(policies))
				for i, policy := range policies {
					out.PolicyIdentifiers[i] = policy.Policy
				}
			}
		}

		if e.Critical {
			return out, UnhandledCriticalExtension{}
		}
	}

	return out, nil
}

// ParseCertificate parses a single certificate from the given ASN.1 DER data.
func ParseCertificate(asn1Data []byte) (*Certificate, error) {
	var cert certificate
	rest, err := asn1.Unmarshal(asn1Data, &cert)
	if err != nil {
		return nil, err
	}
	if len(rest) > 0 {
		return nil, asn1.SyntaxError{"trailing data"}
	}

	return parseCertificate(&cert)
}

// ParseCertificates parses one or more certificates from the given ASN.1 DER
// data. The certificates must be concatenated with no intermediate padding.
func ParseCertificates(asn1Data []byte) ([]*Certificate, error) {
	var v []*certificate

	for len(asn1Data) > 0 {
		cert := new(certificate)
		var err error
		asn1Data, err = asn1.Unmarshal(asn1Data, cert)
		if err != nil {
			return nil, err
		}
		v = append(v, cert)
	}

	ret := make([]*Certificate, len(v))
	for i, ci := range v {
		cert, err := parseCertificate(ci)
		if err != nil {
			return nil, err
		}
		ret[i] = cert
	}

	return ret, nil
}

func reverseBitsInAByte(in byte) byte {
	b1 := in>>4 | in<<4
	b2 := b1>>2&0x33 | b1<<2&0xcc
	b3 := b2>>1&0x55 | b2<<1&0xaa
	return b3
}

var (
	oidExtensionSubjectKeyId        = []int{2, 5, 29, 14}
	oidExtensionKeyUsage            = []int{2, 5, 29, 15}
	oidExtensionAuthorityKeyId      = []int{2, 5, 29, 35}
	oidExtensionBasicConstraints    = []int{2, 5, 29, 19}
	oidExtensionSubjectAltName      = []int{2, 5, 29, 17}
	oidExtensionCertificatePolicies = []int{2, 5, 29, 32}
	oidExtensionNameConstraints     = []int{2, 5, 29, 30}
)

func buildExtensions(template *Certificate) (ret []pkix.Extension, err error) {
	ret = make([]pkix.Extension, 7 /* maximum number of elements. */ )
	n := 0

	if template.KeyUsage != 0 {
		ret[n].Id = oidExtensionKeyUsage
		ret[n].Critical = true

		var a [2]byte
		a[0] = reverseBitsInAByte(byte(template.KeyUsage))
		a[1] = reverseBitsInAByte(byte(template.KeyUsage >> 8))

		l := 1
		if a[1] != 0 {
			l = 2
		}

		ret[n].Value, err = asn1.Marshal(asn1.BitString{Bytes: a[0:l], BitLength: l * 8})
		if err != nil {
			return
		}
		n++
	}

	if template.BasicConstraintsValid {
		ret[n].Id = oidExtensionBasicConstraints
		ret[n].Value, err = asn1.Marshal(basicConstraints{template.IsCA, template.MaxPathLen})
		ret[n].Critical = true
		if err != nil {
			return
		}
		n++
	}

	if len(template.SubjectKeyId) > 0 {
		ret[n].Id = oidExtensionSubjectKeyId
		ret[n].Value, err = asn1.Marshal(template.SubjectKeyId)
		if err != nil {
			return
		}
		n++
	}

	if len(template.AuthorityKeyId) > 0 {
		ret[n].Id = oidExtensionAuthorityKeyId
		ret[n].Value, err = asn1.Marshal(authKeyId{template.AuthorityKeyId})
		if err != nil {
			return
		}
		n++
	}

	if len(template.DNSNames) > 0 {
		ret[n].Id = oidExtensionSubjectAltName
		rawValues := make([]asn1.RawValue, len(template.DNSNames))
		for i, name := range template.DNSNames {
			rawValues[i] = asn1.RawValue{Tag: 2, Class: 2, Bytes: []byte(name)}
		}
		ret[n].Value, err = asn1.Marshal(rawValues)
		if err != nil {
			return
		}
		n++
	}

	if len(template.PolicyIdentifiers) > 0 {
		ret[n].Id = oidExtensionCertificatePolicies
		policies := make([]policyInformation, len(template.PolicyIdentifiers))
		for i, policy := range template.PolicyIdentifiers {
			policies[i].Policy = policy
		}
		ret[n].Value, err = asn1.Marshal(policies)
		if err != nil {
			return
		}
		n++
	}

	if len(template.PermittedDNSDomains) > 0 {
		ret[n].Id = oidExtensionNameConstraints
		ret[n].Critical = template.PermittedDNSDomainsCritical

		var out nameConstraints
		out.Permitted = make([]generalSubtree, len(template.PermittedDNSDomains))
		for i, permitted := range template.PermittedDNSDomains {
			out.Permitted[i] = generalSubtree{Name: permitted}
		}
		ret[n].Value, err = asn1.Marshal(out)
		if err != nil {
			return
		}
		n++
	}

	// Adding another extension here? Remember to update the maximum number
	// of elements in the make() at the top of the function.

	return ret[0:n], nil
}

var (
	oidSHA1WithRSA = []int{1, 2, 840, 113549, 1, 1, 5}
	oidRSA         = []int{1, 2, 840, 113549, 1, 1, 1}
)

// CreateSelfSignedCertificate creates a new certificate based on
// a template. The following members of template are used: SerialNumber,
// Subject, NotBefore, NotAfter, KeyUsage, BasicConstraintsValid, IsCA,
// MaxPathLen, SubjectKeyId, DNSNames, PermittedDNSDomainsCritical,
// PermittedDNSDomains.
//
// The certificate is signed by parent. If parent is equal to template then the
// certificate is self-signed. The parameter pub is the public key of the
// signee and priv is the private key of the signer.
//
// The returned slice is the certificate in DER encoding.
func CreateCertificate(rand io.Reader, template, parent *Certificate, pub *rsa.PublicKey, priv *rsa.PrivateKey) (cert []byte, err error) {
	asn1PublicKey, err := asn1.Marshal(rsaPublicKey{
		N: pub.N,
		E: pub.E,
	})
	if err != nil {
		return
	}

	if len(parent.SubjectKeyId) > 0 {
		template.AuthorityKeyId = parent.SubjectKeyId
	}

	extensions, err := buildExtensions(template)
	if err != nil {
		return
	}

	asn1Issuer, err := asn1.Marshal(parent.Subject.ToRDNSequence())
	if err != nil {
		return
	}
	asn1Subject, err := asn1.Marshal(template.Subject.ToRDNSequence())
	if err != nil {
		return
	}

	encodedPublicKey := asn1.BitString{BitLength: len(asn1PublicKey) * 8, Bytes: asn1PublicKey}
	c := tbsCertificate{
		Version:            2,
		SerialNumber:       template.SerialNumber,
		SignatureAlgorithm: pkix.AlgorithmIdentifier{Algorithm: oidSHA1WithRSA},
		Issuer:             asn1.RawValue{FullBytes: asn1Issuer},
		Validity:           validity{template.NotBefore, template.NotAfter},
		Subject:            asn1.RawValue{FullBytes: asn1Subject},
		PublicKey:          publicKeyInfo{nil, pkix.AlgorithmIdentifier{Algorithm: oidRSA}, encodedPublicKey},
		Extensions:         extensions,
	}

	tbsCertContents, err := asn1.Marshal(c)
	if err != nil {
		return
	}

	c.Raw = tbsCertContents

	h := sha1.New()
	h.Write(tbsCertContents)
	digest := h.Sum()

	signature, err := rsa.SignPKCS1v15(rand, priv, crypto.SHA1, digest)
	if err != nil {
		return
	}

	cert, err = asn1.Marshal(certificate{
		nil,
		c,
		pkix.AlgorithmIdentifier{Algorithm: oidSHA1WithRSA},
		asn1.BitString{Bytes: signature, BitLength: len(signature) * 8},
	})
	return
}

// pemCRLPrefix is the magic string that indicates that we have a PEM encoded
// CRL.
var pemCRLPrefix = []byte("-----BEGIN X509 CRL")
// pemType is the type of a PEM encoded CRL.
var pemType = "X509 CRL"

// ParseCRL parses a CRL from the given bytes. It's often the case that PEM
// encoded CRLs will appear where they should be DER encoded, so this function
// will transparently handle PEM encoding as long as there isn't any leading
// garbage.
func ParseCRL(crlBytes []byte) (certList *pkix.CertificateList, err error) {
	if bytes.HasPrefix(crlBytes, pemCRLPrefix) {
		block, _ := pem.Decode(crlBytes)
		if block != nil && block.Type == pemType {
			crlBytes = block.Bytes
		}
	}
	return ParseDERCRL(crlBytes)
}

// ParseDERCRL parses a DER encoded CRL from the given bytes.
func ParseDERCRL(derBytes []byte) (certList *pkix.CertificateList, err error) {
	certList = new(pkix.CertificateList)
	_, err = asn1.Unmarshal(derBytes, certList)
	if err != nil {
		certList = nil
	}
	return
}

// CreateCRL returns a DER encoded CRL, signed by this Certificate, that
// contains the given list of revoked certificates.
func (c *Certificate) CreateCRL(rand io.Reader, priv *rsa.PrivateKey, revokedCerts []pkix.RevokedCertificate, now, expiry *time.Time) (crlBytes []byte, err error) {
	tbsCertList := pkix.TBSCertificateList{
		Version: 2,
		Signature: pkix.AlgorithmIdentifier{
			Algorithm: oidSignatureSHA1WithRSA,
		},
		Issuer:              c.Subject.ToRDNSequence(),
		ThisUpdate:          now,
		NextUpdate:          expiry,
		RevokedCertificates: revokedCerts,
	}

	tbsCertListContents, err := asn1.Marshal(tbsCertList)
	if err != nil {
		return
	}

	h := sha1.New()
	h.Write(tbsCertListContents)
	digest := h.Sum()

	signature, err := rsa.SignPKCS1v15(rand, priv, crypto.SHA1, digest)
	if err != nil {
		return
	}

	return asn1.Marshal(pkix.CertificateList{
		TBSCertList: tbsCertList,
		SignatureAlgorithm: pkix.AlgorithmIdentifier{
			Algorithm: oidSignatureSHA1WithRSA,
		},
		SignatureValue: asn1.BitString{Bytes: signature, BitLength: len(signature) * 8},
	})
}
