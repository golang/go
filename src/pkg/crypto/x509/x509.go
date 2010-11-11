// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This package parses X.509-encoded keys and certificates.
package x509

import (
	"asn1"
	"big"
	"container/vector"
	"crypto/rsa"
	"crypto/sha1"
	"hash"
	"io"
	"os"
	"strings"
	"time"
)

// pkcs1PrivateKey is a structure which mirrors the PKCS#1 ASN.1 for an RSA private key.
type pkcs1PrivateKey struct {
	Version int
	N       asn1.RawValue
	E       int
	D       asn1.RawValue
	P       asn1.RawValue
	Q       asn1.RawValue
}

// rawValueIsInteger returns true iff the given ASN.1 RawValue is an INTEGER type.
func rawValueIsInteger(raw *asn1.RawValue) bool {
	return raw.Class == 0 && raw.Tag == 2 && raw.IsCompound == false
}

// ParsePKCS1PrivateKey returns an RSA private key from its ASN.1 PKCS#1 DER encoded form.
func ParsePKCS1PrivateKey(der []byte) (key *rsa.PrivateKey, err os.Error) {
	var priv pkcs1PrivateKey
	rest, err := asn1.Unmarshal(der, &priv)
	if len(rest) > 0 {
		err = asn1.SyntaxError{"trailing data"}
		return
	}
	if err != nil {
		return
	}

	if !rawValueIsInteger(&priv.N) ||
		!rawValueIsInteger(&priv.D) ||
		!rawValueIsInteger(&priv.P) ||
		!rawValueIsInteger(&priv.Q) {
		err = asn1.StructuralError{"tags don't match"}
		return
	}

	key = &rsa.PrivateKey{
		PublicKey: rsa.PublicKey{
			E: priv.E,
			N: new(big.Int).SetBytes(priv.N.Bytes),
		},
		D: new(big.Int).SetBytes(priv.D.Bytes),
		P: new(big.Int).SetBytes(priv.P.Bytes),
		Q: new(big.Int).SetBytes(priv.Q.Bytes),
	}

	err = key.Validate()
	if err != nil {
		return nil, err
	}
	return
}

// MarshalPKCS1PrivateKey converts a private key to ASN.1 DER encoded form.
func MarshalPKCS1PrivateKey(key *rsa.PrivateKey) []byte {
	priv := pkcs1PrivateKey{
		Version: 1,
		N:       asn1.RawValue{Tag: 2, Bytes: key.PublicKey.N.Bytes()},
		E:       key.PublicKey.E,
		D:       asn1.RawValue{Tag: 2, Bytes: key.D.Bytes()},
		P:       asn1.RawValue{Tag: 2, Bytes: key.P.Bytes()},
		Q:       asn1.RawValue{Tag: 2, Bytes: key.Q.Bytes()},
	}

	b, _ := asn1.Marshal(priv)
	return b
}

// These structures reflect the ASN.1 structure of X.509 certificates.:

type certificate struct {
	TBSCertificate     tbsCertificate
	SignatureAlgorithm algorithmIdentifier
	SignatureValue     asn1.BitString
}

type tbsCertificate struct {
	Raw                asn1.RawContent
	Version            int "optional,explicit,default:1,tag:0"
	SerialNumber       asn1.RawValue
	SignatureAlgorithm algorithmIdentifier
	Issuer             rdnSequence
	Validity           validity
	Subject            rdnSequence
	PublicKey          publicKeyInfo
	UniqueId           asn1.BitString "optional,tag:1"
	SubjectUniqueId    asn1.BitString "optional,tag:2"
	Extensions         []extension    "optional,explicit,tag:3"
}

type algorithmIdentifier struct {
	Algorithm asn1.ObjectIdentifier
}

type rdnSequence []relativeDistinguishedNameSET

type relativeDistinguishedNameSET []attributeTypeAndValue

type attributeTypeAndValue struct {
	Type  asn1.ObjectIdentifier
	Value interface{}
}

type validity struct {
	NotBefore, NotAfter *time.Time
}

type publicKeyInfo struct {
	Algorithm algorithmIdentifier
	PublicKey asn1.BitString
}

type extension struct {
	Id       asn1.ObjectIdentifier
	Critical bool "optional"
	Value    []byte
}

// RFC 5280,  4.2.1.1
type authKeyId struct {
	Id []byte "optional,tag:0"
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
)

type PublicKeyAlgorithm int

const (
	UnknownPublicKeyAlgorithm PublicKeyAlgorithm = iota
	RSA
)

// Name represents an X.509 distinguished name. This only includes the common
// elements of a DN.  Additional elements in the name are ignored.
type Name struct {
	Country, Organization, OrganizationalUnit []string
	Locality, Province                        []string
	StreetAddress, PostalCode                 []string
	SerialNumber, CommonName                  string
}

func (n *Name) fillFromRDNSequence(rdns *rdnSequence) {
	for _, rdn := range *rdns {
		if len(rdn) == 0 {
			continue
		}
		atv := rdn[0]
		value, ok := atv.Value.(string)
		if !ok {
			continue
		}

		t := atv.Type
		if len(t) == 4 && t[0] == 2 && t[1] == 5 && t[2] == 4 {
			switch t[3] {
			case 3:
				n.CommonName = value
			case 5:
				n.SerialNumber = value
			case 6:
				n.Country = append(n.Country, value)
			case 7:
				n.Locality = append(n.Locality, value)
			case 8:
				n.Province = append(n.Province, value)
			case 9:
				n.StreetAddress = append(n.StreetAddress, value)
			case 10:
				n.Organization = append(n.Organization, value)
			case 11:
				n.OrganizationalUnit = append(n.OrganizationalUnit, value)
			case 17:
				n.PostalCode = append(n.PostalCode, value)
			}
		}
	}
}

var (
	oidCountry            = []int{2, 5, 4, 6}
	oidOrganization       = []int{2, 5, 4, 10}
	oidOrganizationalUnit = []int{2, 5, 4, 11}
	oidCommonName         = []int{2, 5, 4, 3}
	oidSerialNumber       = []int{2, 5, 4, 5}
	oidLocatity           = []int{2, 5, 4, 7}
	oidProvince           = []int{2, 5, 4, 8}
	oidStreetAddress      = []int{2, 5, 4, 9}
	oidPostalCode         = []int{2, 5, 4, 17}
)

// appendRDNs appends a relativeDistinguishedNameSET to the given rdnSequence
// and returns the new value. The relativeDistinguishedNameSET contains an
// attributeTypeAndValue for each of the given values. See RFC 5280, A.1, and
// search for AttributeTypeAndValue.
func appendRDNs(in rdnSequence, values []string, oid asn1.ObjectIdentifier) rdnSequence {
	if len(values) == 0 {
		return in
	}

	s := make([]attributeTypeAndValue, len(values))
	for i, value := range values {
		s[i].Type = oid
		s[i].Value = value
	}

	return append(in, s)
}

func (n Name) toRDNSequence() (ret rdnSequence) {
	ret = appendRDNs(ret, n.Country, oidCountry)
	ret = appendRDNs(ret, n.Organization, oidOrganization)
	ret = appendRDNs(ret, n.OrganizationalUnit, oidOrganizationalUnit)
	ret = appendRDNs(ret, n.Locality, oidLocatity)
	ret = appendRDNs(ret, n.Province, oidProvince)
	ret = appendRDNs(ret, n.StreetAddress, oidStreetAddress)
	ret = appendRDNs(ret, n.PostalCode, oidPostalCode)
	if len(n.CommonName) > 0 {
		ret = appendRDNs(ret, []string{n.CommonName}, oidCommonName)
	}
	if len(n.SerialNumber) > 0 {
		ret = appendRDNs(ret, []string{n.SerialNumber}, oidSerialNumber)
	}

	return ret
}

func getSignatureAlgorithmFromOID(oid []int) SignatureAlgorithm {
	if len(oid) == 7 && oid[0] == 1 && oid[1] == 2 && oid[2] == 840 &&
		oid[3] == 113549 && oid[4] == 1 && oid[5] == 1 {
		switch oid[6] {
		case 2:
			return MD2WithRSA
		case 4:
			return MD5WithRSA
		case 5:
			return SHA1WithRSA
		case 11:
			return SHA256WithRSA
		case 12:
			return SHA384WithRSA
		case 13:
			return SHA512WithRSA
		}
	}

	return UnknownSignatureAlgorithm
}

func getPublicKeyAlgorithmFromOID(oid []int) PublicKeyAlgorithm {
	if len(oid) == 7 && oid[0] == 1 && oid[1] == 2 && oid[2] == 840 &&
		oid[3] == 113549 && oid[4] == 1 && oid[5] == 1 {
		switch oid[6] {
		case 1:
			return RSA
		}
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

// A Certificate represents an X.509 certificate.
type Certificate struct {
	Raw                []byte // Raw ASN.1 DER contents.
	Signature          []byte
	SignatureAlgorithm SignatureAlgorithm

	PublicKeyAlgorithm PublicKeyAlgorithm
	PublicKey          interface{}

	Version             int
	SerialNumber        []byte
	Issuer              Name
	Subject             Name
	NotBefore, NotAfter *time.Time // Validity bounds.
	KeyUsage            KeyUsage

	BasicConstraintsValid bool // if true then the next two fields are valid.
	IsCA                  bool
	MaxPathLen            int

	SubjectKeyId   []byte
	AuthorityKeyId []byte

	// Subject Alternate Name values
	DNSNames       []string
	EmailAddresses []string

	PolicyIdentifiers []asn1.ObjectIdentifier
}

// UnsupportedAlgorithmError results from attempting to perform an operation
// that involves algorithms that are not currently implemented.
type UnsupportedAlgorithmError struct{}

func (UnsupportedAlgorithmError) String() string {
	return "cannot verify signature: algorithm unimplemented"
}

// ConstraintViolationError results when a requested usage is not permitted by
// a certificate. For example: checking a signature when the public key isn't a
// certificate signing key.
type ConstraintViolationError struct{}

func (ConstraintViolationError) String() string {
	return "invalid signature: parent certificate cannot sign this kind of certificate"
}

// CheckSignatureFrom verifies that the signature on c is a valid signature
// from parent.
func (c *Certificate) CheckSignatureFrom(parent *Certificate) (err os.Error) {
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

	var h hash.Hash
	var hashType rsa.PKCS1v15Hash

	switch c.SignatureAlgorithm {
	case SHA1WithRSA:
		h = sha1.New()
		hashType = rsa.HashSHA1
	default:
		return UnsupportedAlgorithmError{}
	}

	pub, ok := parent.PublicKey.(*rsa.PublicKey)
	if !ok {
		return UnsupportedAlgorithmError{}
	}

	h.Write(c.Raw)
	digest := h.Sum()

	return rsa.VerifyPKCS1v15(pub, hashType, digest, c.Signature)
}

func matchHostnames(pattern, host string) bool {
	if len(pattern) == 0 || len(host) == 0 {
		return false
	}

	patternParts := strings.Split(pattern, ".", -1)
	hostParts := strings.Split(host, ".", -1)

	if len(patternParts) != len(hostParts) {
		return false
	}

	for i, patternPart := range patternParts {
		if patternPart == "*" {
			continue
		}
		if patternPart != hostParts[i] {
			return false
		}
	}

	return true
}

type HostnameError struct {
	Certificate *Certificate
	Host        string
}

func (h *HostnameError) String() string {
	var valid string
	c := h.Certificate
	if len(c.DNSNames) > 0 {
		valid = strings.Join(c.DNSNames, ", ")
	} else {
		valid = c.Subject.CommonName
	}
	return "certificate is valid for " + valid + ", not " + h.Host
}

// VerifyHostname returns nil if c is a valid certificate for the named host.
// Otherwise it returns an os.Error describing the mismatch.
func (c *Certificate) VerifyHostname(h string) os.Error {
	if len(c.DNSNames) > 0 {
		for _, match := range c.DNSNames {
			if matchHostnames(match, h) {
				return nil
			}
		}
		// If Subject Alt Name is given, we ignore the common name.
	} else if matchHostnames(c.Subject.CommonName, h) {
		return nil
	}

	return &HostnameError{c, h}
}

type UnhandledCriticalExtension struct{}

func (h UnhandledCriticalExtension) String() string {
	return "unhandled critical extension"
}

type basicConstraints struct {
	IsCA       bool "optional"
	MaxPathLen int  "optional"
}

type rsaPublicKey struct {
	N asn1.RawValue
	E int
}

// RFC 5280 4.2.1.4
type policyInformation struct {
	Policy asn1.ObjectIdentifier
	// policyQualifiers omitted
}

func parsePublicKey(algo PublicKeyAlgorithm, asn1Data []byte) (interface{}, os.Error) {
	switch algo {
	case RSA:
		p := new(rsaPublicKey)
		_, err := asn1.Unmarshal(asn1Data, p)
		if err != nil {
			return nil, err
		}

		if !rawValueIsInteger(&p.N) {
			return nil, asn1.StructuralError{"tags don't match"}
		}

		pub := &rsa.PublicKey{
			E: p.E,
			N: new(big.Int).SetBytes(p.N.Bytes),
		}
		return pub, nil
	default:
		return nil, nil
	}

	panic("unreachable")
}

func parseCertificate(in *certificate) (*Certificate, os.Error) {
	out := new(Certificate)
	out.Raw = in.TBSCertificate.Raw

	out.Signature = in.SignatureValue.RightAlign()
	out.SignatureAlgorithm =
		getSignatureAlgorithmFromOID(in.TBSCertificate.SignatureAlgorithm.Algorithm)

	out.PublicKeyAlgorithm =
		getPublicKeyAlgorithmFromOID(in.TBSCertificate.PublicKey.Algorithm.Algorithm)
	var err os.Error
	out.PublicKey, err = parsePublicKey(out.PublicKeyAlgorithm, in.TBSCertificate.PublicKey.PublicKey.RightAlign())
	if err != nil {
		return nil, err
	}

	out.Version = in.TBSCertificate.Version + 1
	out.SerialNumber = in.TBSCertificate.SerialNumber.Bytes
	out.Issuer.fillFromRDNSequence(&in.TBSCertificate.Issuer)
	out.Subject.fillFromRDNSequence(&in.TBSCertificate.Subject)
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
				var constriants basicConstraints
				_, err := asn1.Unmarshal(e.Value, &constriants)

				if err == nil {
					out.BasicConstraintsValid = true
					out.IsCA = constriants.IsCA
					out.MaxPathLen = constriants.MaxPathLen
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

			case 35:
				// RFC 5280, 4.2.1.1
				var a authKeyId
				_, err = asn1.Unmarshal(e.Value, &a)
				if err != nil {
					return nil, err
				}
				out.AuthorityKeyId = a.Id
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
func ParseCertificate(asn1Data []byte) (*Certificate, os.Error) {
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
func ParseCertificates(asn1Data []byte) ([]*Certificate, os.Error) {
	v := new(vector.Vector)

	for len(asn1Data) > 0 {
		cert := new(certificate)
		var err os.Error
		asn1Data, err = asn1.Unmarshal(asn1Data, cert)
		if err != nil {
			return nil, err
		}
		v.Push(cert)
	}

	ret := make([]*Certificate, v.Len())
	for i := 0; i < v.Len(); i++ {
		cert, err := parseCertificate(v.At(i).(*certificate))
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
)

func buildExtensions(template *Certificate) (ret []extension, err os.Error) {
	ret = make([]extension, 6 /* maximum number of elements. */ )
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
// MaxPathLen, SubjectKeyId, DNSNames.
//
// The certificate is signed by parent. If parent is equal to template then the
// certificate is self-signed. The parameter pub is the public key of the
// signee and priv is the private key of the signer.
//
// The returned slice is the certificate in DER encoding.
func CreateCertificate(rand io.Reader, template, parent *Certificate, pub *rsa.PublicKey, priv *rsa.PrivateKey) (cert []byte, err os.Error) {
	asn1PublicKey, err := asn1.Marshal(rsaPublicKey{
		N: asn1.RawValue{Tag: 2, Bytes: pub.N.Bytes()},
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

	encodedPublicKey := asn1.BitString{BitLength: len(asn1PublicKey) * 8, Bytes: asn1PublicKey}
	c := tbsCertificate{
		Version:            2,
		SerialNumber:       asn1.RawValue{Bytes: template.SerialNumber, Tag: 2},
		SignatureAlgorithm: algorithmIdentifier{oidSHA1WithRSA},
		Issuer:             parent.Subject.toRDNSequence(),
		Validity:           validity{template.NotBefore, template.NotAfter},
		Subject:            template.Subject.toRDNSequence(),
		PublicKey:          publicKeyInfo{algorithmIdentifier{oidRSA}, encodedPublicKey},
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

	signature, err := rsa.SignPKCS1v15(rand, priv, rsa.HashSHA1, digest)
	if err != nil {
		return
	}

	cert, err = asn1.Marshal(certificate{
		c,
		algorithmIdentifier{oidSHA1WithRSA},
		asn1.BitString{Bytes: signature, BitLength: len(signature) * 8},
	})
	return
}
