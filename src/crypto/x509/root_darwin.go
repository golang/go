// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !ios
// +build !ios

package x509

import (
	"bytes"
	macOS "crypto/x509/internal/macos"
	"fmt"
	"os"
	"strings"
)

var debugDarwinRoots = strings.Contains(os.Getenv("GODEBUG"), "x509roots=1")

func (c *Certificate) systemVerify(opts *VerifyOptions) (chains [][]*Certificate, err error) {
	return nil, nil
}

func loadSystemRoots() (*CertPool, error) {
	var trustedRoots []*Certificate
	untrustedRoots := make(map[string]bool)

	// macOS has three trust domains: one for CAs added by users to their
	// "login" keychain, one for CAs added by Admins to the "System" keychain,
	// and one for the CAs that ship with the OS.
	for _, domain := range []macOS.SecTrustSettingsDomain{
		macOS.SecTrustSettingsDomainUser,
		macOS.SecTrustSettingsDomainAdmin,
		macOS.SecTrustSettingsDomainSystem,
	} {
		certs, err := macOS.SecTrustSettingsCopyCertificates(domain)
		if err == macOS.ErrNoTrustSettings {
			continue
		} else if err != nil {
			return nil, err
		}
		defer macOS.CFRelease(certs)

		for i := 0; i < macOS.CFArrayGetCount(certs); i++ {
			c := macOS.CFArrayGetValueAtIndex(certs, i)
			cert, err := exportCertificate(c)
			if err != nil {
				if debugDarwinRoots {
					fmt.Fprintf(os.Stderr, "crypto/x509: domain %d, certificate #%d: %v\n", domain, i, err)
				}
				continue
			}

			var result macOS.SecTrustSettingsResult
			if domain == macOS.SecTrustSettingsDomainSystem {
				// Certs found in the system domain are always trusted. If the user
				// configures "Never Trust" on such a cert, it will also be found in the
				// admin or user domain, causing it to be added to untrustedRoots.
				result = macOS.SecTrustSettingsResultTrustRoot
			} else {
				result, err = sslTrustSettingsResult(c)
				if err != nil {
					if debugDarwinRoots {
						fmt.Fprintf(os.Stderr, "crypto/x509: trust settings for %v: %v\n", cert.Subject, err)
					}
					continue
				}
				if debugDarwinRoots {
					fmt.Fprintf(os.Stderr, "crypto/x509: trust settings for %v: %d\n", cert.Subject, result)
				}
			}

			switch result {
			// "Note the distinction between the results kSecTrustSettingsResultTrustRoot
			// and kSecTrustSettingsResultTrustAsRoot: The former can only be applied to
			// root (self-signed) certificates; the latter can only be applied to
			// non-root certificates."
			case macOS.SecTrustSettingsResultTrustRoot:
				if isRootCertificate(cert) {
					trustedRoots = append(trustedRoots, cert)
				}
			case macOS.SecTrustSettingsResultTrustAsRoot:
				if !isRootCertificate(cert) {
					trustedRoots = append(trustedRoots, cert)
				}

			case macOS.SecTrustSettingsResultDeny:
				// Add this certificate to untrustedRoots, which are subtracted
				// from trustedRoots, so that we don't have to evaluate policies
				// for every root in the system domain, but still apply user and
				// admin policies that override system roots.
				untrustedRoots[string(cert.Raw)] = true

			case macOS.SecTrustSettingsResultUnspecified:
				// Certificates with unspecified trust should be added to a pool
				// of intermediates for chain building, but we don't support it
				// at the moment. This is Issue 35631.

			default:
				if debugDarwinRoots {
					fmt.Fprintf(os.Stderr, "crypto/x509: unknown trust setting for %v: %d\n", cert.Subject, result)
				}
			}
		}
	}

	pool := NewCertPool()
	for _, cert := range trustedRoots {
		if !untrustedRoots[string(cert.Raw)] {
			pool.AddCert(cert)
		}
	}
	return pool, nil
}

// exportCertificate returns a *Certificate for a SecCertificateRef.
func exportCertificate(cert macOS.CFRef) (*Certificate, error) {
	data, err := macOS.SecItemExport(cert)
	if err != nil {
		return nil, err
	}
	defer macOS.CFRelease(data)
	der := macOS.CFDataToSlice(data)

	return ParseCertificate(der)
}

// isRootCertificate reports whether Subject and Issuer match.
func isRootCertificate(cert *Certificate) bool {
	return bytes.Equal(cert.RawSubject, cert.RawIssuer)
}

// sslTrustSettingsResult obtains the final kSecTrustSettingsResult value for a
// certificate in the user or admin domain, combining usage constraints for the
// SSL SecTrustSettingsPolicy,
//
// It ignores SecTrustSettingsKeyUsage and kSecTrustSettingsAllowedError, and
// doesn't support kSecTrustSettingsDefaultRootCertSetting.
//
// https://developer.apple.com/documentation/security/1400261-sectrustsettingscopytrustsetting
func sslTrustSettingsResult(cert macOS.CFRef) (macOS.SecTrustSettingsResult, error) {
	// In Apple's implementation user trust settings override admin trust settings
	// (which themselves override system trust settings). If SecTrustSettingsCopyTrustSettings
	// fails, or returns a NULL trust settings, when looking for the user trust
	// settings then fallback to checking the admin trust settings.
	//
	// See Security-59306.41.2/trust/headers/SecTrustSettings.h for a description of
	// the trust settings overrides, and SecLegacyAnchorSourceCopyUsageConstraints in
	// Security-59306.41.2/trust/trustd/SecCertificateSource.c for a concrete example
	// of how Apple applies the override in the case of NULL trust settings, or non
	// success errors.
	trustSettings, err := macOS.SecTrustSettingsCopyTrustSettings(cert, macOS.SecTrustSettingsDomainUser)
	if err != nil || trustSettings == 0 {
		if debugDarwinRoots && err != macOS.ErrNoTrustSettings {
			fmt.Fprintf(os.Stderr, "crypto/x509: SecTrustSettingsCopyTrustSettings for SecTrustSettingsDomainUser failed: %s\n", err)
		}
		trustSettings, err = macOS.SecTrustSettingsCopyTrustSettings(cert, macOS.SecTrustSettingsDomainAdmin)
	}
	if err != nil || trustSettings == 0 {
		// If there are neither user nor admin trust settings for a certificate returned
		// from SecTrustSettingsCopyCertificates Apple returns kSecTrustSettingsResultInvalid,
		// as this method is intended to return certificates _which have trust settings_.
		// The most likely case for this being triggered is that the existing trust settings
		// are invalid and cannot be properly parsed. In this case SecTrustSettingsCopyTrustSettings
		// returns errSecInvalidTrustSettings. The existing cgo implementation returns
		// kSecTrustSettingsResultUnspecified in this case, which mostly matches the Apple
		// implementation because we don't do anything with certificates marked with this
		// result.
		//
		// See SecPVCGetTrustSettingsResult in Security-59306.41.2/trust/trustd/SecPolicyServer.c
		if debugDarwinRoots && err != macOS.ErrNoTrustSettings {
			fmt.Fprintf(os.Stderr, "crypto/x509: SecTrustSettingsCopyTrustSettings for SecTrustSettingsDomainAdmin failed: %s\n", err)
		}
		return macOS.SecTrustSettingsResultUnspecified, nil
	}
	defer macOS.CFRelease(trustSettings)

	// "An empty trust settings array means 'always trust this certificate' with an
	// overall trust setting for the certificate of kSecTrustSettingsResultTrustRoot."
	if macOS.CFArrayGetCount(trustSettings) == 0 {
		return macOS.SecTrustSettingsResultTrustRoot, nil
	}

	isSSLPolicy := func(policyRef macOS.CFRef) bool {
		properties := macOS.SecPolicyCopyProperties(policyRef)
		defer macOS.CFRelease(properties)
		if v, ok := macOS.CFDictionaryGetValueIfPresent(properties, macOS.SecPolicyOid); ok {
			return macOS.CFEqual(v, macOS.CFRef(macOS.SecPolicyAppleSSL))
		}
		return false
	}

	for i := 0; i < macOS.CFArrayGetCount(trustSettings); i++ {
		tSetting := macOS.CFArrayGetValueAtIndex(trustSettings, i)

		// First, check if this trust setting is constrained to a non-SSL policy.
		if policyRef, ok := macOS.CFDictionaryGetValueIfPresent(tSetting, macOS.SecTrustSettingsPolicy); ok {
			if !isSSLPolicy(policyRef) {
				continue
			}
		}

		// Then check if it is restricted to a hostname, so not a root.
		if _, ok := macOS.CFDictionaryGetValueIfPresent(tSetting, macOS.SecTrustSettingsPolicyString); ok {
			continue
		}

		cfNum, ok := macOS.CFDictionaryGetValueIfPresent(tSetting, macOS.SecTrustSettingsResultKey)
		// "If this key is not present, a default value of kSecTrustSettingsResultTrustRoot is assumed."
		if !ok {
			return macOS.SecTrustSettingsResultTrustRoot, nil
		}
		result, err := macOS.CFNumberGetValue(cfNum)
		if err != nil {
			return 0, err
		}

		// If multiple dictionaries match, we are supposed to "OR" them,
		// the semantics of which are not clear. Since TrustRoot and TrustAsRoot
		// are mutually exclusive, Deny should probably override, and Invalid and
		// Unspecified be overridden, approximate this by stopping at the first
		// TrustRoot, TrustAsRoot or Deny.
		switch r := macOS.SecTrustSettingsResult(result); r {
		case macOS.SecTrustSettingsResultTrustRoot,
			macOS.SecTrustSettingsResultTrustAsRoot,
			macOS.SecTrustSettingsResultDeny:
			return r, nil
		}
	}

	// If trust settings are present, but none of them match the policy...
	// the docs don't tell us what to do.
	//
	// "Trust settings for a given use apply if any of the dictionaries in the
	// certificate’s trust settings array satisfies the specified use." suggests
	// that it's as if there were no trust settings at all, so we should maybe
	// fallback to the admin trust settings? TODO(golang.org/issue/38888).

	return macOS.SecTrustSettingsResultUnspecified, nil
}
