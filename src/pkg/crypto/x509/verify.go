// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x509

import (
	"strings"
	"time"
	"unicode/utf8"
)

type InvalidReason int

const (
	// NotAuthorizedToSign results when a certificate is signed by another
	// which isn't marked as a CA certificate.
	NotAuthorizedToSign InvalidReason = iota
	// Expired results when a certificate has expired, based on the time
	// given in the VerifyOptions.
	Expired
	// CANotAuthorizedForThisName results when an intermediate or root
	// certificate has a name constraint which doesn't include the name
	// being checked.
	CANotAuthorizedForThisName
)

// CertificateInvalidError results when an odd error occurs. Users of this
// library probably want to handle all these errors uniformly.
type CertificateInvalidError struct {
	Cert   *Certificate
	Reason InvalidReason
}

func (e CertificateInvalidError) Error() string {
	switch e.Reason {
	case NotAuthorizedToSign:
		return "x509: certificate is not authorized to sign other other certificates"
	case Expired:
		return "x509: certificate has expired or is not yet valid"
	case CANotAuthorizedForThisName:
		return "x509: a root or intermediate certificate is not authorized to sign in this domain"
	}
	return "x509: unknown error"
}

// HostnameError results when the set of authorized names doesn't match the
// requested name.
type HostnameError struct {
	Certificate *Certificate
	Host        string
}

func (h HostnameError) Error() string {
	var valid string
	c := h.Certificate
	if len(c.DNSNames) > 0 {
		valid = strings.Join(c.DNSNames, ", ")
	} else {
		valid = c.Subject.CommonName
	}
	return "certificate is valid for " + valid + ", not " + h.Host
}

// UnknownAuthorityError results when the certificate issuer is unknown
type UnknownAuthorityError struct {
	cert *Certificate
}

func (e UnknownAuthorityError) Error() string {
	return "x509: certificate signed by unknown authority"
}

// VerifyOptions contains parameters for Certificate.Verify. It's a structure
// because other PKIX verification APIs have ended up needing many options.
type VerifyOptions struct {
	DNSName       string
	Intermediates *CertPool
	Roots         *CertPool
	CurrentTime   time.Time // if zero, the current time is used
}

const (
	leafCertificate = iota
	intermediateCertificate
	rootCertificate
)

// isValid performs validity checks on the c.
func (c *Certificate) isValid(certType int, opts *VerifyOptions) error {
	now := opts.CurrentTime
	if now.IsZero() {
		now = time.Now()
	}
	if now.Before(c.NotBefore) || now.After(c.NotAfter) {
		return CertificateInvalidError{c, Expired}
	}

	if len(c.PermittedDNSDomains) > 0 {
		for _, domain := range c.PermittedDNSDomains {
			if opts.DNSName == domain ||
				(strings.HasSuffix(opts.DNSName, domain) &&
					len(opts.DNSName) >= 1+len(domain) &&
					opts.DNSName[len(opts.DNSName)-len(domain)-1] == '.') {
				continue
			}

			return CertificateInvalidError{c, CANotAuthorizedForThisName}
		}
	}

	// KeyUsage status flags are ignored. From Engineering Security, Peter
	// Gutmann: A European government CA marked its signing certificates as
	// being valid for encryption only, but no-one noticed. Another
	// European CA marked its signature keys as not being valid for
	// signatures. A different CA marked its own trusted root certificate
	// as being invalid for certificate signing.  Another national CA
	// distributed a certificate to be used to encrypt data for the
	// country’s tax authority that was marked as only being usable for
	// digital signatures but not for encryption. Yet another CA reversed
	// the order of the bit flags in the keyUsage due to confusion over
	// encoding endianness, essentially setting a random keyUsage in
	// certificates that it issued. Another CA created a self-invalidating
	// certificate by adding a certificate policy statement stipulating
	// that the certificate had to be used strictly as specified in the
	// keyUsage, and a keyUsage containing a flag indicating that the RSA
	// encryption key could only be used for Diffie-Hellman key agreement.

	if certType == intermediateCertificate && (!c.BasicConstraintsValid || !c.IsCA) {
		return CertificateInvalidError{c, NotAuthorizedToSign}
	}

	return nil
}

// Verify attempts to verify c by building one or more chains from c to a
// certificate in opts.roots, using certificates in opts.Intermediates if
// needed. If successful, it returns one or chains where the first element of
// the chain is c and the last element is from opts.Roots.
//
// WARNING: this doesn't do any revocation checking.
func (c *Certificate) Verify(opts VerifyOptions) (chains [][]*Certificate, err error) {
	err = c.isValid(leafCertificate, &opts)
	if err != nil {
		return
	}
	if len(opts.DNSName) > 0 {
		err = c.VerifyHostname(opts.DNSName)
		if err != nil {
			return
		}
	}
	return c.buildChains(make(map[int][][]*Certificate), []*Certificate{c}, &opts)
}

func appendToFreshChain(chain []*Certificate, cert *Certificate) []*Certificate {
	n := make([]*Certificate, len(chain)+1)
	copy(n, chain)
	n[len(chain)] = cert
	return n
}

func (c *Certificate) buildChains(cache map[int][][]*Certificate, currentChain []*Certificate, opts *VerifyOptions) (chains [][]*Certificate, err error) {
	for _, rootNum := range opts.Roots.findVerifiedParents(c) {
		root := opts.Roots.certs[rootNum]
		err = root.isValid(rootCertificate, opts)
		if err != nil {
			continue
		}
		chains = append(chains, appendToFreshChain(currentChain, root))
	}

nextIntermediate:
	for _, intermediateNum := range opts.Intermediates.findVerifiedParents(c) {
		intermediate := opts.Intermediates.certs[intermediateNum]
		for _, cert := range currentChain {
			if cert == intermediate {
				continue nextIntermediate
			}
		}
		err = intermediate.isValid(intermediateCertificate, opts)
		if err != nil {
			continue
		}
		var childChains [][]*Certificate
		childChains, ok := cache[intermediateNum]
		if !ok {
			childChains, err = intermediate.buildChains(cache, appendToFreshChain(currentChain, intermediate), opts)
			cache[intermediateNum] = childChains
		}
		chains = append(chains, childChains...)
	}

	if len(chains) > 0 {
		err = nil
	}

	if len(chains) == 0 && err == nil {
		err = UnknownAuthorityError{c}
	}

	return
}

func matchHostnames(pattern, host string) bool {
	if len(pattern) == 0 || len(host) == 0 {
		return false
	}

	patternParts := strings.Split(pattern, ".")
	hostParts := strings.Split(host, ".")

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

// toLowerCaseASCII returns a lower-case version of in. See RFC 6125 6.4.1. We use
// an explicitly ASCII function to avoid any sharp corners resulting from
// performing Unicode operations on DNS labels.
func toLowerCaseASCII(in string) string {
	// If the string is already lower-case then there's nothing to do.
	isAlreadyLowerCase := true
	for _, c := range in {
		if c == utf8.RuneError {
			// If we get a UTF-8 error then there might be
			// upper-case ASCII bytes in the invalid sequence.
			isAlreadyLowerCase = false
			break
		}
		if 'A' <= c && c <= 'Z' {
			isAlreadyLowerCase = false
			break
		}
	}

	if isAlreadyLowerCase {
		return in
	}

	out := []byte(in)
	for i, c := range out {
		if 'A' <= c && c <= 'Z' {
			out[i] += 'a' - 'A'
		}
	}
	return string(out)
}

// VerifyHostname returns nil if c is a valid certificate for the named host.
// Otherwise it returns an error describing the mismatch.
func (c *Certificate) VerifyHostname(h string) error {
	lowered := toLowerCaseASCII(h)

	if len(c.DNSNames) > 0 {
		for _, match := range c.DNSNames {
			if matchHostnames(toLowerCaseASCII(match), lowered) {
				return nil
			}
		}
		// If Subject Alt Name is given, we ignore the common name.
	} else if matchHostnames(toLowerCaseASCII(c.Subject.CommonName), lowered) {
		return nil
	}

	return HostnameError{c, h}
}
