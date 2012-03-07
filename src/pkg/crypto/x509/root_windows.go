// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x509

import (
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
			if err != nil {
				return nil, err
			}

			err = syscall.CertFreeCertificateContext(ctx)
			if err != nil {
				return nil, err
			}
		}
	}

	return storeCtx, nil
}

func (c *Certificate) systemVerify(opts *VerifyOptions) (chains [][]*Certificate, err error) {
	hasDNSName := opts != nil && len(opts.DNSName) > 0

	storeCtx, err := createStoreContext(c, opts)
	if err != nil {
		return nil, err
	}
	defer syscall.CertFreeCertificateContext(storeCtx)

	para := new(syscall.CertChainPara)
	para.Size = uint32(unsafe.Sizeof(*para))
	para.RequestedUsage.Type = syscall.USAGE_MATCH_TYPE_AND

	// If there's a DNSName set in opts, assume we're verifying
	// a certificate from a TLS server.
	if hasDNSName {
		oids := []*byte{&syscall.OID_PKIX_KP_SERVER_AUTH[0]}
		para.RequestedUsage.Usage.Length = uint32(len(oids))
		para.RequestedUsage.Usage.UsageIdentifiers = &oids[0]
	} else {
		para.RequestedUsage.Usage.Length = 0
		para.RequestedUsage.Usage.UsageIdentifiers = nil
	}

	var verifyTime *syscall.Filetime
	if opts != nil && !opts.CurrentTime.IsZero() {
		ft := syscall.NsecToFiletime(opts.CurrentTime.UnixNano())
		verifyTime = &ft
	}

	// CertGetCertificateChain will traverse Windows's root stores
	// in an attempt to build a verified certificate chain.  Once
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

	if chainCtx.TrustStatus.ErrorStatus != syscall.CERT_TRUST_NO_ERROR {
		status := chainCtx.TrustStatus.ErrorStatus
		switch status {
		case syscall.CERT_TRUST_IS_NOT_TIME_VALID:
			return nil, CertificateInvalidError{c, Expired}
		default:
			return nil, UnknownAuthorityError{c}
		}
	}

	simpleChains := (*[1 << 20]*syscall.CertSimpleChain)(unsafe.Pointer(chainCtx.Chains))[:]
	if chainCtx.ChainCount == 0 {
		return nil, UnknownAuthorityError{c}
	}
	verifiedChain := simpleChains[int(chainCtx.ChainCount)-1]

	elements := (*[1 << 20]*syscall.CertChainElement)(unsafe.Pointer(verifiedChain.Elements))[:]
	if verifiedChain.NumElements == 0 {
		return nil, UnknownAuthorityError{c}
	}

	var chain []*Certificate
	for i := 0; i < int(verifiedChain.NumElements); i++ {
		// Copy the buf, since ParseCertificate does not create its own copy.
		cert := elements[i].CertContext
		encodedCert := (*[1 << 20]byte)(unsafe.Pointer(cert.EncodedCert))[:]
		buf := make([]byte, cert.Length)
		copy(buf, encodedCert[:])
		parsedCert, err := ParseCertificate(buf)
		if err != nil {
			return nil, err
		}
		chain = append(chain, parsedCert)
	}

	// Apply the system SSL policy if VerifyOptions dictates that we
	// must check for a DNS name.
	if hasDNSName {
		sslPara := &syscall.SSLExtraCertChainPolicyPara{
			AuthType:   syscall.AUTHTYPE_SERVER,
			ServerName: syscall.StringToUTF16Ptr(opts.DNSName),
		}
		sslPara.Size = uint32(unsafe.Sizeof(*sslPara))

		para := &syscall.CertChainPolicyPara{
			ExtraPolicyPara: uintptr(unsafe.Pointer(sslPara)),
		}
		para.Size = uint32(unsafe.Sizeof(*para))

		status := syscall.CertChainPolicyStatus{}
		err = syscall.CertVerifyCertificateChainPolicy(syscall.CERT_CHAIN_POLICY_SSL, chainCtx, para, &status)
		if err != nil {
			return nil, err
		}

		if status.Error != 0 {
			switch status.Error {
			case syscall.CERT_E_EXPIRED:
				return nil, CertificateInvalidError{c, Expired}
			case syscall.CERT_E_CN_NO_MATCH:
				return nil, HostnameError{c, opts.DNSName}
			case syscall.CERT_E_UNTRUSTEDROOT:
				return nil, UnknownAuthorityError{c}
			default:
				return nil, UnknownAuthorityError{c}
			}
		}
	}

	chains = make([][]*Certificate, 1)
	chains[0] = chain

	return chains, nil
}

func initSystemRoots() {
	systemRoots = NewCertPool()
}
