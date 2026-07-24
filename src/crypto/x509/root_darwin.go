// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x509

import (
	"crypto/x509/internal/macos"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// macOS has no default SSL_CERT_{FILE,DIR} paths.
var certFiles, certDirectories []string

// secPolicyMu serializes SecPolicyCreateSSL against repairExecutableDir
// cleanup so concurrent verifiers cannot remove a directory another call
// still needs.
var secPolicyMu sync.Mutex

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
	// SecPolicyCreateSSL returns NULL when the directory containing this
	// process's executable is missing (macOS Security.framework quirk).
	// That commonly happens with "go run" after the go command deletes its
	// temporary build directory while a child process is still running.
	// The directory must exist during the SecPolicyCreateSSL call; we recreate
	// it temporarily if needed and remove any empty directories we created.
	// See go.dev/issue/68557 and go.dev/issue/54590.
	secPolicyMu.Lock()
	cleanup := repairExecutableDir()
	sslPolicy, err := macos.SecPolicyCreateSSL(opts.DNSName)
	cleanup()
	secPolicyMu.Unlock()
	if err != nil {
		return nil, fmt.Errorf("x509: %w (the directory containing this executable may be missing; see https://go.dev/issue/68557)", err)
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

// repairExecutableDir recreates the directory containing the running
// executable if it is missing. The returned cleanup removes any empty
// directories that were created, deepest first.
func repairExecutableDir() (cleanup func()) {
	cleanup = func() {}
	exe, err := os.Executable()
	if err != nil {
		return cleanup
	}
	dir := filepath.Dir(exe)
	if fi, err := os.Stat(dir); err == nil && fi.IsDir() {
		return cleanup
	}

	// Record missing ancestors so cleanup only removes what we create.
	var missing []string
	for cur := dir; ; {
		fi, err := os.Stat(cur)
		if err == nil {
			if !fi.IsDir() {
				return cleanup
			}
			break
		}
		missing = append(missing, cur)
		parent := filepath.Dir(cur)
		if parent == cur {
			break
		}
		cur = parent
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return cleanup
	}
	return func() {
		for _, d := range missing {
			entries, err := os.ReadDir(d)
			if err != nil || len(entries) > 0 {
				return
			}
			if err := os.Remove(d); err != nil {
				return
			}
		}
	}
}
