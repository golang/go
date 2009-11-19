// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This package parses X.509-encoded keys and certificates.
package x509

import (
	"asn1";
	"big";
	"container/vector";
	"crypto/rsa";
	"crypto/sha1";
	"hash";
	"os";
	"strings";
	"time";
)

// pkcs1PrivateKey is a structure which mirrors the PKCS#1 ASN.1 for an RSA private key.
type pkcs1PrivateKey struct {
	Version	int;
	N	asn1.RawValue;
	E	int;
	D	asn1.RawValue;
	P	asn1.RawValue;
	Q	asn1.RawValue;
}

// rawValueIsInteger returns true iff the given ASN.1 RawValue is an INTEGER type.
func rawValueIsInteger(raw *asn1.RawValue) bool {
	return raw.Class == 0 && raw.Tag == 2 && raw.IsCompound == false
}

// ParsePKCS1PrivateKey returns an RSA private key from its ASN.1 PKCS#1 DER encoded form.
func ParsePKCS1PrivateKey(der []byte) (key *rsa.PrivateKey, err os.Error) {
	var priv pkcs1PrivateKey;
	rest, err := asn1.Unmarshal(&priv, der);
	if len(rest) > 0 {
		err = asn1.SyntaxError{"trailing data"};
		return;
	}
	if err != nil {
		return
	}

	if !rawValueIsInteger(&priv.N) ||
		!rawValueIsInteger(&priv.D) ||
		!rawValueIsInteger(&priv.P) ||
		!rawValueIsInteger(&priv.Q) {
		err = asn1.StructuralError{"tags don't match"};
		return;
	}

	key = &rsa.PrivateKey{
		PublicKey: rsa.PublicKey{
			E: priv.E,
			N: new(big.Int).SetBytes(priv.N.Bytes),
		},
		D: new(big.Int).SetBytes(priv.D.Bytes),
		P: new(big.Int).SetBytes(priv.P.Bytes),
		Q: new(big.Int).SetBytes(priv.Q.Bytes),
	};

	err = key.Validate();
	if err != nil {
		return nil, err
	}
	return;
}

// These structures reflect the ASN.1 structure of X.509 certificates.:

type certificate struct {
	TBSCertificate		tbsCertificate;
	SignatureAlgorithm	algorithmIdentifier;
	SignatureValue		asn1.BitString;
}

type tbsCertificate struct {
	Raw			asn1.RawContents;
	Version			int	"optional,explicit,default:1,tag:0";
	SerialNumber		asn1.RawValue;
	SignatureAlgorithm	algorithmIdentifier;
	Issuer			rdnSequence;
	Validity		validity;
	Subject			rdnSequence;
	PublicKey		publicKeyInfo;
	UniqueId		asn1.BitString	"optional,explicit,tag:1";
	SubjectUniqueId		asn1.BitString	"optional,explicit,tag:2";
	Extensions		[]extension	"optional,explicit,tag:3";
}

type algorithmIdentifier struct {
	Algorithm asn1.ObjectIdentifier;
}

type rdnSequence []relativeDistinguishedName

type relativeDistinguishedName []attributeTypeAndValue

type attributeTypeAndValue struct {
	Type	asn1.ObjectIdentifier;
	Value	interface{};
}

type validity struct {
	NotBefore, NotAfter *time.Time;
}

type publicKeyInfo struct {
	Algorithm	algorithmIdentifier;
	PublicKey	asn1.BitString;
}

type extension struct {
	Id		asn1.ObjectIdentifier;
	Critical	bool	"optional";
	Value		[]byte;
}

// RFC 5280,  4.2.1.1
type authKeyId struct {
	Id []byte "optional,tag:0";
}

type SignatureAlgorithm int

const (
	UnknownSignatureAlgorithm	SignatureAlgorithm	= iota;
	MD2WithRSA;
	MD5WithRSA;
	SHA1WithRSA;
	SHA256WithRSA;
	SHA384WithRSA;
	SHA512WithRSA;
)

type PublicKeyAlgorithm int

const (
	UnknownPublicKeyAlgorithm	PublicKeyAlgorithm	= iota;
	RSA;
)

// Name represents an X.509 distinguished name. This only includes the common
// elements of a DN.  Additional elements in the name are ignored.
type Name struct {
	Country, Organization, OrganizationalUnit	string;
	CommonName, SerialNumber, Locality		string;
	Province, StreetAddress, PostalCode		string;
}

func (n *Name) fillFromRDNSequence(rdns *rdnSequence) {
	for _, rdn := range *rdns {
		if len(rdn) == 0 {
			continue
		}
		atv := rdn[0];
		value, ok := atv.Value.(string);
		if !ok {
			continue
		}

		t := atv.Type;
		if len(t) == 4 && t[0] == 2 && t[1] == 5 && t[2] == 4 {
			switch t[3] {
			case 3:
				n.CommonName = value
			case 5:
				n.SerialNumber = value
			case 6:
				n.Country = value
			case 7:
				n.Locality = value
			case 8:
				n.Province = value
			case 9:
				n.StreetAddress = value
			case 10:
				n.Organization = value
			case 11:
				n.OrganizationalUnit = value
			case 17:
				n.PostalCode = value
			}
		}
	}
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

	return UnknownSignatureAlgorithm;
}

func getPublicKeyAlgorithmFromOID(oid []int) PublicKeyAlgorithm {
	if len(oid) == 7 && oid[0] == 1 && oid[1] == 2 && oid[2] == 840 &&
		oid[3] == 113549 && oid[4] == 1 && oid[5] == 1 {
		switch oid[6] {
		case 1:
			return RSA
		}
	}

	return UnknownPublicKeyAlgorithm;
}

// KeyUsage represents the set of actions that are valid for a given key. It's
// a bitmap of the KeyUsage* constants.
type KeyUsage int

const (
	KeyUsageDigitalSignature	KeyUsage	= 1 << iota;
	KeyUsageContentCommitment;
	KeyUsageKeyEncipherment;
	KeyUsageDataEncipherment;
	KeyUsageKeyAgreement;
	KeyUsageCertSign;
	KeyUsageCRLSign;
	KeyUsageEncipherOnly;
	KeyUsageDecipherOnly;
)

// A Certificate represents an X.509 certificate.
type Certificate struct {
	Raw			[]byte;	// Raw ASN.1 DER contents.
	Signature		[]byte;
	SignatureAlgorithm	SignatureAlgorithm;

	PublicKeyAlgorithm	PublicKeyAlgorithm;
	PublicKey		interface{};

	Version			int;
	SerialNumber		[]byte;
	Issuer			Name;
	Subject			Name;
	NotBefore, NotAfter	*time.Time;	// Validity bounds.
	KeyUsage		KeyUsage;

	BasicConstraintsValid	bool;	// if true then the next two fields are valid.
	IsCA			bool;
	MaxPathLen		int;

	SubjectKeyId	[]byte;
	AuthorityKeyId	[]byte;

	// Subject Alternate Name values
	DNSNames	[]string;
	EmailAddresses	[]string;
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

	var h hash.Hash;
	var hashType rsa.PKCS1v15Hash;

	switch c.SignatureAlgorithm {
	case SHA1WithRSA:
		h = sha1.New();
		hashType = rsa.HashSHA1;
	default:
		return UnsupportedAlgorithmError{}
	}

	pub, ok := parent.PublicKey.(*rsa.PublicKey);
	if !ok {
		return UnsupportedAlgorithmError{}
	}

	h.Write(c.Raw);
	digest := h.Sum();

	return rsa.VerifyPKCS1v15(pub, hashType, digest, c.Signature);
}

func matchHostnames(pattern, host string) bool {
	if len(pattern) == 0 || len(host) == 0 {
		return false
	}

	patternParts := strings.Split(pattern, ".", 0);
	hostParts := strings.Split(host, ".", 0);

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

	return true;
}

// IsValidForHost returns true iff c is a valid certificate for the given host.
func (c *Certificate) IsValidForHost(h string) bool {
	if len(c.DNSNames) > 0 {
		for _, match := range c.DNSNames {
			if matchHostnames(match, h) {
				return true
			}
		}
		// If Subject Alt Name is given, we ignore the common name.
		return false;
	}

	return matchHostnames(c.Subject.CommonName, h);
}

type UnhandledCriticalExtension struct{}

func (h UnhandledCriticalExtension) String() string {
	return "unhandled critical extension"
}

type basicConstraints struct {
	IsCA		bool	"optional";
	MaxPathLen	int	"optional";
}

type rsaPublicKey struct {
	N	asn1.RawValue;
	E	int;
}

func parsePublicKey(algo PublicKeyAlgorithm, asn1Data []byte) (interface{}, os.Error) {
	switch algo {
	case RSA:
		p := new(rsaPublicKey);
		_, err := asn1.Unmarshal(p, asn1Data);
		if err != nil {
			return nil, err
		}

		if !rawValueIsInteger(&p.N) {
			return nil, asn1.StructuralError{"tags don't match"}
		}

		pub := &rsa.PublicKey{
			E: p.E,
			N: new(big.Int).SetBytes(p.N.Bytes),
		};
		return pub, nil;
	default:
		return nil, nil
	}

	panic("unreachable");
}

func appendString(in []string, v string) (out []string) {
	if cap(in)-len(in) < 1 {
		out = make([]string, len(in)+1, len(in)*2+1);
		for i, v := range in {
			out[i] = v
		}
	} else {
		out = in[0 : len(in)+1]
	}
	out[len(in)] = v;
	return out;
}

func parseCertificate(in *certificate) (*Certificate, os.Error) {
	out := new(Certificate);
	out.Raw = in.TBSCertificate.Raw;

	out.Signature = in.SignatureValue.RightAlign();
	out.SignatureAlgorithm =
		getSignatureAlgorithmFromOID(in.TBSCertificate.SignatureAlgorithm.Algorithm);

	out.PublicKeyAlgorithm =
		getPublicKeyAlgorithmFromOID(in.TBSCertificate.PublicKey.Algorithm.Algorithm);
	var err os.Error;
	out.PublicKey, err = parsePublicKey(out.PublicKeyAlgorithm, in.TBSCertificate.PublicKey.PublicKey.RightAlign());
	if err != nil {
		return nil, err
	}

	out.Version = in.TBSCertificate.Version;
	out.SerialNumber = in.TBSCertificate.SerialNumber.Bytes;
	out.Issuer.fillFromRDNSequence(&in.TBSCertificate.Issuer);
	out.Subject.fillFromRDNSequence(&in.TBSCertificate.Subject);
	out.NotBefore = in.TBSCertificate.Validity.NotBefore;
	out.NotAfter = in.TBSCertificate.Validity.NotAfter;

	for _, e := range in.TBSCertificate.Extensions {
		if len(e.Id) == 4 && e.Id[0] == 2 && e.Id[1] == 5 && e.Id[2] == 29 {
			switch e.Id[3] {
			case 15:
				// RFC 5280, 4.2.1.3
				var usageBits asn1.BitString;
				_, err := asn1.Unmarshal(&usageBits, e.Value);

				if err == nil {
					var usage int;
					for i := 0; i < 9; i++ {
						if usageBits.At(i) != 0 {
							usage |= 1 << uint(i)
						}
					}
					out.KeyUsage = KeyUsage(usage);
					continue;
				}
			case 19:
				// RFC 5280, 4.2.1.9
				var constriants basicConstraints;
				_, err := asn1.Unmarshal(&constriants, e.Value);

				if err == nil {
					out.BasicConstraintsValid = true;
					out.IsCA = constriants.IsCA;
					out.MaxPathLen = constriants.MaxPathLen;
					continue;
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
				var seq asn1.RawValue;
				_, err := asn1.Unmarshal(&seq, e.Value);
				if err != nil {
					return nil, err
				}
				if !seq.IsCompound || seq.Tag != 16 || seq.Class != 0 {
					return nil, asn1.StructuralError{"bad SAN sequence"}
				}

				parsedName := false;

				rest := seq.Bytes;
				for len(rest) > 0 {
					var v asn1.RawValue;
					rest, err = asn1.Unmarshal(&v, rest);
					if err != nil {
						return nil, err
					}
					switch v.Tag {
					case 1:
						out.EmailAddresses = appendString(out.EmailAddresses, string(v.Bytes));
						parsedName = true;
					case 2:
						out.DNSNames = appendString(out.DNSNames, string(v.Bytes));
						parsedName = true;
					}
				}

				if parsedName {
					continue
				}
				// If we didn't parse any of the names then we
				// fall through to the critical check below.

			case 35:
				// RFC 5280, 4.2.1.1
				var a authKeyId;
				_, err = asn1.Unmarshal(&a, e.Value);
				if err != nil {
					return nil, err
				}
				out.AuthorityKeyId = a.Id;
				continue;

			case 14:
				// RFC 5280, 4.2.1.2
				out.SubjectKeyId = e.Value;
				continue;
			}
		}

		if e.Critical {
			return out, UnhandledCriticalExtension{}
		}
	}

	return out, nil;
}

// ParseCertificate parses a single certificate from the given ASN.1 DER data.
func ParseCertificate(asn1Data []byte) (*Certificate, os.Error) {
	var cert certificate;
	rest, err := asn1.Unmarshal(&cert, asn1Data);
	if err != nil {
		return nil, err
	}
	if len(rest) > 0 {
		return nil, asn1.SyntaxError{"trailing data"}
	}

	return parseCertificate(&cert);
}

// ParseCertificates parses one or more certificates from the given ASN.1 DER
// data. The certificates must be concatenated with no intermediate padding.
func ParseCertificates(asn1Data []byte) ([]*Certificate, os.Error) {
	v := vector.New(0);

	for len(asn1Data) > 0 {
		cert := new(certificate);
		var err os.Error;
		asn1Data, err = asn1.Unmarshal(cert, asn1Data);
		if err != nil {
			return nil, err
		}
		v.Push(cert);
	}

	ret := make([]*Certificate, v.Len());
	for i := 0; i < v.Len(); i++ {
		cert, err := parseCertificate(v.At(i).(*certificate));
		if err != nil {
			return nil, err
		}
		ret[i] = cert;
	}

	return ret, nil;
}
