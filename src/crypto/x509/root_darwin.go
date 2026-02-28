// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x509

import (
	"crypto/x509/internal/macos"
	"errors"
	"fmt"
)

func (c *Certificate) systemVerify(opts *VerifyOptions) (chains [][]*Certificate, err error) {
	certs := macos.CFArrayCreateMutable()
	defer macos.ReleaseCFArray(certs)
	leaf, err := macos.SecCertificateCreateWithData(c.Raw)
	if err != nil {
		return nil, errors.New("invalid leaf certificate")
	}
	macos.CFArrayAppendValue(certs, leaf)
	if opts.Intermediates != nil {
		for _, lc := range opts.Intermediates.lazyCerts {
			c, err := lc.getCert()
			if err != nil {
				return nil, err
			}
			sc, err := macos.SecCertificateCreateWithData(c.Raw)
			if err != nil {
				return nil, err
			}
			macos.CFArrayAppendValue(certs, sc)
		}
	}

	policies := macos.CFArrayCreateMutable()
	defer macos.ReleaseCFArray(policies)
	sslPolicy, err := macos.SecPolicyCreateSSL(opts.DNSName)
	if err != nil {
		return nil, err
	}
	macos.CFArrayAppendValue(policies, sslPolicy)

	trustObj, err := macos.SecTrustCreateWithCertificates(certs, policies)
	if err != nil {
		return nil, err
	}
	defer macos.CFRelease(trustObj)

	if !opts.CurrentTime.IsZero() {
		dateRef := macos.TimeToCFDateRef(opts.CurrentTime)
		defer macos.CFRelease(dateRef)
		if err := macos.SecTrustSetVerifyDate(trustObj, dateRef); err != nil {
			return nil, err
		}
	}

	// TODO(roland): we may want to allow passing in SCTs via VerifyOptions and
	// set them via SecTrustSetSignedCertificateTimestamps, since Apple will
	// always enforce its SCT requirements, and there are still _some_ people
	// using TLS or OCSP for that.

	if ret, err := macos.SecTrustEvaluateWithError(trustObj); err != nil {
		switch ret {
		case macos.ErrSecCertificateExpired:
			return nil, CertificateInvalidError{c, Expired, err.Error()}
		case macos.ErrSecHostNameMismatch:
			return nil, HostnameError{c, opts.DNSName}
		case macos.ErrSecNotTrusted:
			return nil, UnknownAuthorityError{Cert: c}
		default:
			return nil, fmt.Errorf("x509: %s", err)
		}
	}

	chain := [][]*Certificate{{}}
	chainRef, err := macos.SecTrustCopyCertificateChain(trustObj)
	if err != nil {
		return nil, err
	}
	defer macos.CFRelease(chainRef)
	for i := 0; i < macos.CFArrayGetCount(chainRef); i++ {
		certRef := macos.CFArrayGetValueAtIndex(chainRef, i)
		cert, err := exportCertificate(certRef)
		if err != nil {
			return nil, err
		}
		chain[0] = append(chain[0], cert)
	}
	if len(chain[0]) == 0 {
		// This should _never_ happen, but to be safe
		return nil, errors.New("x509: macos certificate verification internal error")
	}

	if opts.DNSName != "" {
		// If we have a DNS name, apply our own name verification
		if err := chain[0][0].VerifyHostname(opts.DNSName); err != nil {
			return nil, err
		}
	}

	keyUsages := opts.KeyUsages
	if len(keyUsages) == 0 {
		keyUsages = []ExtKeyUsage{ExtKeyUsageServerAuth}
	}

	// If any key usage is acceptable then we're done.
	for _, usage := range keyUsages {
		if usage == ExtKeyUsageAny {
			return chain, nil
		}
	}

	if !checkChainForKeyUsage(chain[0], keyUsages) {
		return nil, CertificateInvalidError{c, IncompatibleUsage, ""}
	}

	return chain, nil
}

// exportCertificate returns a *Certificate for a SecCertificateRef.
func exportCertificate(cert macos.CFRef) (*Certificate, error) {
	data, err := macos.SecCertificateCopyData(cert)
	if err != nil {
		return nil, err
	}
	return ParseCertificate(data)
}

func loadSystemRoots() (*CertPool, error) {
	return &CertPool{systemPool: true}, nil
}
