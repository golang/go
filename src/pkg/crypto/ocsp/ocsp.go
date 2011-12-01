// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package ocsp parses OCSP responses as specified in RFC 2560. OCSP responses
// are signed messages attesting to the validity of a certificate for a small
// period of time. This is used to manage revocation for X.509 certificates.
package ocsp

import (
	"crypto"
	"crypto/rsa"
	_ "crypto/sha1"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/asn1"
	"time"
)

var idPKIXOCSPBasic = asn1.ObjectIdentifier([]int{1, 3, 6, 1, 5, 5, 7, 48, 1, 1})
var idSHA1WithRSA = asn1.ObjectIdentifier([]int{1, 2, 840, 113549, 1, 1, 5})

// These are internal structures that reflect the ASN.1 structure of an OCSP
// response. See RFC 2560, section 4.2.

const (
	ocspSuccess       = 0
	ocspMalformed     = 1
	ocspInternalError = 2
	ocspTryLater      = 3
	ocspSigRequired   = 4
	ocspUnauthorized  = 5
)

type certID struct {
	HashAlgorithm pkix.AlgorithmIdentifier
	NameHash      []byte
	IssuerKeyHash []byte
	SerialNumber  asn1.RawValue
}

type responseASN1 struct {
	Status   asn1.Enumerated
	Response responseBytes `asn1:"explicit,tag:0"`
}

type responseBytes struct {
	ResponseType asn1.ObjectIdentifier
	Response     []byte
}

type basicResponse struct {
	TBSResponseData    responseData
	SignatureAlgorithm pkix.AlgorithmIdentifier
	Signature          asn1.BitString
	Certificates       []asn1.RawValue `asn1:"explicit,tag:0,optional"`
}

type responseData struct {
	Raw           asn1.RawContent
	Version       int              `asn1:"optional,default:1,explicit,tag:0"`
	RequestorName pkix.RDNSequence `asn1:"optional,explicit,tag:1"`
	KeyHash       []byte           `asn1:"optional,explicit,tag:2"`
	ProducedAt    time.Time
	Responses     []singleResponse
}

type singleResponse struct {
	CertID     certID
	Good       asn1.Flag   `asn1:"explicit,tag:0,optional"`
	Revoked    revokedInfo `asn1:"explicit,tag:1,optional"`
	Unknown    asn1.Flag   `asn1:"explicit,tag:2,optional"`
	ThisUpdate time.Time
	NextUpdate time.Time `asn1:"explicit,tag:0,optional"`
}

type revokedInfo struct {
	RevocationTime time.Time
	Reason         int `asn1:"explicit,tag:0,optional"`
}

// This is the exposed reflection of the internal OCSP structures.

const (
	// Good means that the certificate is valid.
	Good = iota
	// Revoked means that the certificate has been deliberately revoked.
	Revoked = iota
	// Unknown means that the OCSP responder doesn't know about the certificate.
	Unknown = iota
	// ServerFailed means that the OCSP responder failed to process the request.
	ServerFailed = iota
)

// Response represents an OCSP response. See RFC 2560.
type Response struct {
	// Status is one of {Good, Revoked, Unknown, ServerFailed}
	Status                                        int
	SerialNumber                                  []byte
	ProducedAt, ThisUpdate, NextUpdate, RevokedAt time.Time
	RevocationReason                              int
	Certificate                                   *x509.Certificate
}

// ParseError results from an invalid OCSP response.
type ParseError string

func (p ParseError) Error() string {
	return string(p)
}

// ParseResponse parses an OCSP response in DER form. It only supports
// responses for a single certificate and only those using RSA signatures.
// Non-RSA responses will result in an x509.UnsupportedAlgorithmError.
// Signature errors or parse failures will result in a ParseError.
func ParseResponse(bytes []byte) (*Response, error) {
	var resp responseASN1
	rest, err := asn1.Unmarshal(bytes, &resp)
	if err != nil {
		return nil, err
	}
	if len(rest) > 0 {
		return nil, ParseError("trailing data in OCSP response")
	}

	ret := new(Response)
	if resp.Status != ocspSuccess {
		ret.Status = ServerFailed
		return ret, nil
	}

	if !resp.Response.ResponseType.Equal(idPKIXOCSPBasic) {
		return nil, ParseError("bad OCSP response type")
	}

	var basicResp basicResponse
	rest, err = asn1.Unmarshal(resp.Response.Response, &basicResp)
	if err != nil {
		return nil, err
	}

	if len(basicResp.Certificates) != 1 {
		return nil, ParseError("OCSP response contains bad number of certificates")
	}

	if len(basicResp.TBSResponseData.Responses) != 1 {
		return nil, ParseError("OCSP response contains bad number of responses")
	}

	ret.Certificate, err = x509.ParseCertificate(basicResp.Certificates[0].FullBytes)
	if err != nil {
		return nil, err
	}

	if ret.Certificate.PublicKeyAlgorithm != x509.RSA || !basicResp.SignatureAlgorithm.Algorithm.Equal(idSHA1WithRSA) {
		return nil, x509.UnsupportedAlgorithmError{}
	}

	hashType := crypto.SHA1
	h := hashType.New()

	pub := ret.Certificate.PublicKey.(*rsa.PublicKey)
	h.Write(basicResp.TBSResponseData.Raw)
	digest := h.Sum(nil)
	signature := basicResp.Signature.RightAlign()

	if rsa.VerifyPKCS1v15(pub, hashType, digest, signature) != nil {
		return nil, ParseError("bad OCSP signature")
	}

	r := basicResp.TBSResponseData.Responses[0]

	ret.SerialNumber = r.CertID.SerialNumber.Bytes

	switch {
	case bool(r.Good):
		ret.Status = Good
	case bool(r.Unknown):
		ret.Status = Unknown
	default:
		ret.Status = Revoked
		ret.RevokedAt = r.Revoked.RevocationTime
		ret.RevocationReason = r.Revoked.Reason
	}

	ret.ProducedAt = basicResp.TBSResponseData.ProducedAt
	ret.ThisUpdate = r.ThisUpdate
	ret.NextUpdate = r.NextUpdate

	return ret, nil
}
