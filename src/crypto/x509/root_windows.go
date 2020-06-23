// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x509

import (
	"errors"
	"syscall"
	"unsafe"
)

// Creates a new *syscall.CertContext representing the leaf certificate in an in-memory
// certificate store containing itself and all of the intermediate certificates specified
// in the opts.Intermediates CertPool.
//
// A pointer to the in-memory store is available in the returned CertContext's Store field.
// The store is automatically freed when the CertContext is freed using
// syscall.CertFreeCertificateContext.
func createStoreContext(leaf *Certificate, opts *VerifyOptions) (*syscall.CertContext, error) {
	var storeCtx *syscall.CertContext

	leafCtx, err := syscall.CertCreateCertificateContext(syscall.X509_ASN_ENCODING|syscall.PKCS_7_ASN_ENCODING, &leaf.Raw[0], uint32(len(leaf.Raw)))
	if err != nil {
		return nil, err
	}
	defer syscall.CertFreeCertificateContext(leafCtx)

	handle, err := syscall.CertOpenStore(syscall.CERT_STORE_PROV_MEMORY, 0, 0, syscall.CERT_STORE_DEFER_CLOSE_UNTIL_LAST_FREE_FLAG, 0)
	if err != nil {
		return nil, err
	}
	defer syscall.CertCloseStore(handle, 0)

	err = syscall.CertAddCertificateContextToStore(handle, leafCtx, syscall.CERT_STORE_ADD_ALWAYS, &storeCtx)
	if err != nil {
		return nil, err
	}

	if opts.Intermediates != nil {
		for _, intermediate := range opts.Intermediates.certs {
			ctx, err := syscall.CertCreateCertificateContext(syscall.X509_ASN_ENCODING|syscall.PKCS_7_ASN_ENCODING, &intermediate.Raw[0], uint32(len(intermediate.Raw)))
			if err != nil {
				return nil, err
			}

			err = syscall.CertAddCertificateContextToStore(handle, ctx, syscall.CERT_STORE_ADD_ALWAYS, nil)
			syscall.CertFreeCertificateContext(ctx)
			if err != nil {
				return nil, err
			}
		}
	}

	return storeCtx, nil
}

// extractSimpleChain extracts the final certificate chain from a CertSimpleChain.
func extractSimpleChain(simpleChain **syscall.CertSimpleChain, count int) (chain []*Certificate, err error) {
	if simpleChain == nil || count == 0 {
		return nil, errors.New("x509: invalid simple chain")
	}

	simpleChains := (*[1 << 20]*syscall.CertSimpleChain)(unsafe.Pointer(simpleChain))[:count:count]
	lastChain := simpleChains[count-1]
	elements := (*[1 << 20]*syscall.CertChainElement)(unsafe.Pointer(lastChain.Elements))[:lastChain.NumElements:lastChain.NumElements]
	for i := 0; i < int(lastChain.NumElements); i++ {
		// Copy the buf, since ParseCertificate does not create its own copy.
		cert := elements[i].CertContext
		encodedCert := (*[1 << 20]byte)(unsafe.Pointer(cert.EncodedCert))[:cert.Length:cert.Length]
		buf := make([]byte, cert.Length)
		copy(buf, encodedCert)
		parsedCert, err := ParseCertificate(buf)
		if err != nil {
			return nil, err
		}
		chain = append(chain, parsedCert)
	}

	return chain, nil
}

// checkChainTrustStatus checks the trust status of the certificate chain, translating
// any errors it finds into Go errors in the process.
func checkChainTrustStatus(c *Certificate, chainCtx *syscall.CertChainContext) error {
	if chainCtx.TrustStatus.ErrorStatus != syscall.CERT_TRUST_NO_ERROR {
		status := chainCtx.TrustStatus.ErrorStatus
		switch status {
		case syscall.CERT_TRUST_IS_NOT_TIME_VALID:
			return CertificateInvalidError{c, Expired, ""}
		default:
			return UnknownAuthorityError{c, nil, nil}
		}
	}
	return nil
}

// checkChainSSLServerPolicy checks that the certificate chain in chainCtx is valid for
// use as a certificate chain for a SSL/TLS server.
func checkChainSSLServerPolicy(c *Certificate, chainCtx *syscall.CertChainContext, opts *VerifyOptions) error {
	servernamep, err := syscall.UTF16PtrFromString(opts.DNSName)
	if err != nil {
		return err
	}
	sslPara := &syscall.SSLExtraCertChainPolicyPara{
		AuthType:   syscall.AUTHTYPE_SERVER,
		ServerName: servernamep,
	}
	sslPara.Size = uint32(unsafe.Sizeof(*sslPara))

	para := &syscall.CertChainPolicyPara{
		ExtraPolicyPara: (syscall.Pointer)(unsafe.Pointer(sslPara)),
	}
	para.Size = uint32(unsafe.Sizeof(*para))

	status := syscall.CertChainPolicyStatus{}
	err = syscall.CertVerifyCertificateChainPolicy(syscall.CERT_CHAIN_POLICY_SSL, chainCtx, para, &status)
	if err != nil {
		return err
	}

	// TODO(mkrautz): use the lChainIndex and lElementIndex fields
	// of the CertChainPolicyStatus to provide proper context, instead
	// using c.
	if status.Error != 0 {
		switch status.Error {
		case syscall.CERT_E_EXPIRED:
			return CertificateInvalidError{c, Expired, ""}
		case syscall.CERT_E_CN_NO_MATCH:
			return HostnameError{c, opts.DNSName}
		case syscall.CERT_E_UNTRUSTEDROOT:
			return UnknownAuthorityError{c, nil, nil}
		default:
			return UnknownAuthorityError{c, nil, nil}
		}
	}

	return nil
}

// systemVerify is like Verify, except that it uses CryptoAPI calls
// to build certificate chains and verify them.
func (c *Certificate) systemVerify(opts *VerifyOptions) (chains [][]*Certificate, err error) {
	hasDNSName := opts != nil && len(opts.DNSName) > 0

	storeCtx, err := createStoreContext(c, opts)
	if err != nil {
		return nil, err
	}
	defer syscall.CertFreeCertificateContext(storeCtx)

	para := new(syscall.CertChainPara)
	para.Size = uint32(unsafe.Sizeof(*para))

	// If there's a DNSName set in opts, assume we're verifying
	// a certificate from a TLS server.
	if hasDNSName {
		oids := []*byte{
			&syscall.OID_PKIX_KP_SERVER_AUTH[0],
			// Both IE and Chrome allow certificates with
			// Server Gated Crypto as well. Some certificates
			// in the wild require them.
			&syscall.OID_SERVER_GATED_CRYPTO[0],
			&syscall.OID_SGC_NETSCAPE[0],
		}
		para.RequestedUsage.Type = syscall.USAGE_MATCH_TYPE_OR
		para.RequestedUsage.Usage.Length = uint32(len(oids))
		para.RequestedUsage.Usage.UsageIdentifiers = &oids[0]
	} else {
		para.RequestedUsage.Type = syscall.USAGE_MATCH_TYPE_AND
		para.RequestedUsage.Usage.Length = 0
		para.RequestedUsage.Usage.UsageIdentifiers = nil
	}

	var verifyTime *syscall.Filetime
	if opts != nil && !opts.CurrentTime.IsZero() {
		ft := syscall.NsecToFiletime(opts.CurrentTime.UnixNano())
		verifyTime = &ft
	}

	// CertGetCertificateChain will traverse Windows's root stores
	// in an attempt to build a verified certificate chain. Once
	// it has found a verified chain, it stops. MSDN docs on
	// CERT_CHAIN_CONTEXT:
	//
	//   When a CERT_CHAIN_CONTEXT is built, the first simple chain
	//   begins with an end certificate and ends with a self-signed
	//   certificate. If that self-signed certificate is not a root
	//   or otherwise trusted certificate, an attempt is made to
	//   build a new chain. CTLs are used to create the new chain
	//   beginning with the self-signed certificate from the original
	//   chain as the end certificate of the new chain. This process
	//   continues building additional simple chains until the first
	//   self-signed certificate is a trusted certificate or until
	//   an additional simple chain cannot be built.
	//
	// The result is that we'll only get a single trusted chain to
	// return to our caller.
	var chainCtx *syscall.CertChainContext
	err = syscall.CertGetCertificateChain(syscall.Handle(0), storeCtx, verifyTime, storeCtx.Store, para, 0, 0, &chainCtx)
	if err != nil {
		return nil, err
	}
	defer syscall.CertFreeCertificateChain(chainCtx)

	err = checkChainTrustStatus(c, chainCtx)
	if err != nil {
		return nil, err
	}

	if hasDNSName {
		err = checkChainSSLServerPolicy(c, chainCtx, opts)
		if err != nil {
			return nil, err
		}
	}

	chain, err := extractSimpleChain(chainCtx.Chains, int(chainCtx.ChainCount))
	if err != nil {
		return nil, err
	}
	if len(chain) < 1 {
		return nil, errors.New("x509: internal error: system verifier returned an empty chain")
	}

	// Mitigate CVE-2020-0601, where the Windows system verifier might be
	// tricked into using custom curve parameters for a trusted root, by
	// double-checking all ECDSA signatures. If the system was tricked into
	// using spoofed parameters, the signature will be invalid for the correct
	// ones we parsed. (We don't support custom curves ourselves.)
	for i, parent := range chain[1:] {
		if parent.PublicKeyAlgorithm != ECDSA {
			continue
		}
		if err := parent.CheckSignature(chain[i].SignatureAlgorithm,
			chain[i].RawTBSCertificate, chain[i].Signature); err != nil {
			return nil, err
		}
	}

	return [][]*Certificate{chain}, nil
}

func loadSystemRoots() (*CertPool, error) {
	// TODO: restore this functionality on Windows. We tried to do
	// it in Go 1.8 but had to revert it. See Issue 18609.
	// Returning (nil, nil) was the old behavior, prior to CL 30578.
	// The if statement here avoids vet complaining about
	// unreachable code below.
	if true {
		return nil, nil
	}

	const CRYPT_E_NOT_FOUND = 0x80092004

	store, err := syscall.CertOpenSystemStore(0, syscall.StringToUTF16Ptr("ROOT"))
	if err != nil {
		return nil, err
	}
	defer syscall.CertCloseStore(store, 0)

	roots := NewCertPool()
	var cert *syscall.CertContext
	for {
		cert, err = syscall.CertEnumCertificatesInStore(store, cert)
		if err != nil {
			if errno, ok := err.(syscall.Errno); ok {
				if errno == CRYPT_E_NOT_FOUND {
					break
				}
			}
			return nil, err
		}
		if cert == nil {
			break
		}
		// Copy the buf, since ParseCertificate does not create its own copy.
		buf := (*[1 << 20]byte)(unsafe.Pointer(cert.EncodedCert))[:cert.Length:cert.Length]
		buf2 := make([]byte, cert.Length)
		copy(buf2, buf)
		if c, err := ParseCertificate(buf2); err == nil {
			roots.AddCert(c)
		}
	}
	return roots, nil
}
