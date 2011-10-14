// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package pkix contains shared, low level structures used for ASN.1 parsing
// and serialization of X.509 certificates, CRL and OCSP.
package pkix

import (
	"asn1"
	"big"
	"time"
)

// AlgorithmIdentifier represents the ASN.1 structure of the same name. See RFC
// 5280, section 4.1.1.2.
type AlgorithmIdentifier struct {
	Algorithm  asn1.ObjectIdentifier
	Parameters asn1.RawValue `asn1:"optional"`
}

type RDNSequence []RelativeDistinguishedNameSET

type RelativeDistinguishedNameSET []AttributeTypeAndValue

type AttributeTypeAndValue struct {
	Type  asn1.ObjectIdentifier
	Value interface{}
}

// Extension represents the ASN.1 structure of the same name. See RFC
// 5280, section 4.2.
type Extension struct {
	Id       asn1.ObjectIdentifier
	Critical bool `asn1:"optional"`
	Value    []byte
}

// Name represents an X.509 distinguished name. This only includes the common
// elements of a DN.  Additional elements in the name are ignored.
type Name struct {
	Country, Organization, OrganizationalUnit []string
	Locality, Province                        []string
	StreetAddress, PostalCode                 []string
	SerialNumber, CommonName                  string

	Names []AttributeTypeAndValue
}

func (n *Name) FillFromRDNSequence(rdns *RDNSequence) {
	for _, rdn := range *rdns {
		if len(rdn) == 0 {
			continue
		}
		atv := rdn[0]
		n.Names = append(n.Names, atv)
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
	oidLocality           = []int{2, 5, 4, 7}
	oidProvince           = []int{2, 5, 4, 8}
	oidStreetAddress      = []int{2, 5, 4, 9}
	oidPostalCode         = []int{2, 5, 4, 17}
)

// appendRDNs appends a relativeDistinguishedNameSET to the given RDNSequence
// and returns the new value. The relativeDistinguishedNameSET contains an
// attributeTypeAndValue for each of the given values. See RFC 5280, A.1, and
// search for AttributeTypeAndValue.
func appendRDNs(in RDNSequence, values []string, oid asn1.ObjectIdentifier) RDNSequence {
	if len(values) == 0 {
		return in
	}

	s := make([]AttributeTypeAndValue, len(values))
	for i, value := range values {
		s[i].Type = oid
		s[i].Value = value
	}

	return append(in, s)
}

func (n Name) ToRDNSequence() (ret RDNSequence) {
	ret = appendRDNs(ret, n.Country, oidCountry)
	ret = appendRDNs(ret, n.Organization, oidOrganization)
	ret = appendRDNs(ret, n.OrganizationalUnit, oidOrganizationalUnit)
	ret = appendRDNs(ret, n.Locality, oidLocality)
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

// CertificateList represents the ASN.1 structure of the same name. See RFC
// 5280, section 5.1. Use Certificate.CheckCRLSignature to verify the
// signature.
type CertificateList struct {
	TBSCertList        TBSCertificateList
	SignatureAlgorithm AlgorithmIdentifier
	SignatureValue     asn1.BitString
}

// HasExpired returns true iff currentTimeSeconds is past the expiry time of
// certList.
func (certList *CertificateList) HasExpired(currentTimeSeconds int64) bool {
	return certList.TBSCertList.NextUpdate.Seconds() <= currentTimeSeconds
}

// TBSCertificateList represents the ASN.1 structure of the same name. See RFC
// 5280, section 5.1.
type TBSCertificateList struct {
	Raw                 asn1.RawContent
	Version             int `asn1:"optional,default:2"`
	Signature           AlgorithmIdentifier
	Issuer              RDNSequence
	ThisUpdate          *time.Time
	NextUpdate          *time.Time
	RevokedCertificates []RevokedCertificate `asn1:"optional"`
	Extensions          []Extension          `asn1:"tag:0,optional,explicit"`
}

// RevokedCertificate represents the ASN.1 structure of the same name. See RFC
// 5280, section 5.1.
type RevokedCertificate struct {
	SerialNumber   *big.Int
	RevocationTime *time.Time
	Extensions     []Extension `asn1:"optional"`
}
