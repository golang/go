// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x509

import (
	macOS "crypto/x509/internal/macos"
	"errors"
)

func (c *Certificate) systemVerify(opts *VerifyOptions) (chains [][]*Certificate, err error) {
	certs := macOS.CFArrayCreateMutable()
	defer macOS.ReleaseCFArray(certs)
	leaf := macOS.SecCertificateCreateWithData(c.Raw)
	macOS.CFArrayAppendValue(certs, leaf)
	if opts.Intermediates != nil {
		for _, lc := range opts.Intermediates.lazyCerts {
			c, err := lc.getCert()
			if err != nil {
				return nil, err
			}
			sc := macOS.SecCertificateCreateWithData(c.Raw)
			macOS.CFArrayAppendValue(certs, sc)
		}
	}

	policies := macOS.CFArrayCreateMutable()
	defer macOS.ReleaseCFArray(policies)
	sslPolicy := macOS.SecPolicyCreateSSL(opts.DNSName)
	macOS.CFArrayAppendValue(policies, sslPolicy)

	trustObj, err := macOS.SecTrustCreateWithCertificates(certs, policies)
	if err != nil {
		return nil, err
	}
	defer macOS.CFRelease(trustObj)

	if !opts.CurrentTime.IsZero() {
		dateRef := macOS.TimeToCFDateRef(opts.CurrentTime)
		defer macOS.CFRelease(dateRef)
		if err := macOS.SecTrustSetVerifyDate(trustObj, dateRef); err != nil {
			return nil, err
		}
	}

	// TODO(roland): we may want to allow passing in SCTs via VerifyOptions and
	// set them via SecTrustSetSignedCertificateTimestamps, since Apple will
	// always enforce its SCT requirements, and there are still _some_ people
	// using TLS or OCSP for that.

	if err := macOS.SecTrustEvaluateWithError(trustObj); err != nil {
		return nil, err
	}

	chain := [][]*Certificate{{}}
	numCerts := macOS.SecTrustGetCertificateCount(trustObj)
	for i := 0; i < numCerts; i++ {
		certRef := macOS.SecTrustGetCertificateAtIndex(trustObj, i)
		cert, err := exportCertificate(certRef)
		if err != nil {
			return nil, err
		}
		chain[0] = append(chain[0], cert)
	}
	if len(chain[0]) == 0 {
		// This should _never_ happen, but to be safe
		return nil, errors.New("x509: macOS certificate verification internal error")
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
func exportCertificate(cert macOS.CFRef) (*Certificate, error) {
	data, err := macOS.SecCertificateCopyData(cert)
	if err != nil {
		return nil, err
	}
	return ParseCertificate(data)
}

func loadSystemRoots() (*CertPool, error) {
	return &CertPool{systemPool: true}, nil
}
