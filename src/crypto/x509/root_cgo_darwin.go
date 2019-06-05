// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build cgo,!arm,!arm64,!ios

package x509

/*
#cgo CFLAGS: -mmacosx-version-min=10.10 -D__MAC_OS_X_VERSION_MAX_ALLOWED=101300
#cgo LDFLAGS: -framework CoreFoundation -framework Security

#include <errno.h>
#include <sys/sysctl.h>

#include <CoreFoundation/CoreFoundation.h>
#include <Security/Security.h>

static Boolean isSSLPolicy(SecPolicyRef policyRef) {
	if (!policyRef) {
		return false;
	}
	CFDictionaryRef properties = SecPolicyCopyProperties(policyRef);
	if (properties == NULL) {
		return false;
	}
	Boolean isSSL = false;
	CFTypeRef value = NULL;
	if (CFDictionaryGetValueIfPresent(properties, kSecPolicyOid, (const void **)&value)) {
		isSSL = CFEqual(value, kSecPolicyAppleSSL);
	}
	CFRelease(properties);
	return isSSL;
}

// sslTrustSettingsResult obtains the final kSecTrustSettingsResult value
// for a certificate in the user or admin domain, combining usage constraints
// for the SSL SecTrustSettingsPolicy, ignoring SecTrustSettingsKeyUsage and
// kSecTrustSettingsAllowedError.
// https://developer.apple.com/documentation/security/1400261-sectrustsettingscopytrustsetting
static SInt32 sslTrustSettingsResult(SecCertificateRef cert) {
	CFArrayRef trustSettings = NULL;
	OSStatus err = SecTrustSettingsCopyTrustSettings(cert, kSecTrustSettingsDomainUser, &trustSettings);

	// According to Apple's SecTrustServer.c, "user trust settings overrule admin trust settings",
	// but the rules of the override are unclear. Let's assume admin trust settings are applicable
	// if and only if user trust settings fail to load or are NULL.
	if (err != errSecSuccess || trustSettings == NULL) {
		if (trustSettings != NULL) CFRelease(trustSettings);
		err = SecTrustSettingsCopyTrustSettings(cert, kSecTrustSettingsDomainAdmin, &trustSettings);
	}

	// > no trust settings [...] means "this certificate must be verified to a known trusted certificate”
	// (Should this cause a fallback from user to admin domain? It's unclear.)
	if (err != errSecSuccess || trustSettings == NULL) {
		if (trustSettings != NULL) CFRelease(trustSettings);
		return kSecTrustSettingsResultUnspecified;
	}

	// > An empty trust settings array means "always trust this certificate” with an
	// > overall trust setting for the certificate of kSecTrustSettingsResultTrustRoot.
	if (CFArrayGetCount(trustSettings) == 0) {
		CFRelease(trustSettings);
		return kSecTrustSettingsResultTrustRoot;
	}

	// kSecTrustSettingsResult is defined as CFSTR("kSecTrustSettingsResult"),
	// but the Go linker's internal linking mode can't handle CFSTR relocations.
	// Create our own dynamic string instead and release it below.
	CFStringRef _kSecTrustSettingsResult = CFStringCreateWithCString(
		NULL, "kSecTrustSettingsResult", kCFStringEncodingUTF8);
	CFStringRef _kSecTrustSettingsPolicy = CFStringCreateWithCString(
		NULL, "kSecTrustSettingsPolicy", kCFStringEncodingUTF8);
	CFStringRef _kSecTrustSettingsPolicyString = CFStringCreateWithCString(
		NULL, "kSecTrustSettingsPolicyString", kCFStringEncodingUTF8);

	CFIndex m; SInt32 result = 0;
	for (m = 0; m < CFArrayGetCount(trustSettings); m++) {
		CFDictionaryRef tSetting = (CFDictionaryRef)CFArrayGetValueAtIndex(trustSettings, m);

		// First, check if this trust setting is constrained to a non-SSL policy.
		SecPolicyRef policyRef;
		if (CFDictionaryGetValueIfPresent(tSetting, _kSecTrustSettingsPolicy, (const void**)&policyRef)) {
			if (!isSSLPolicy(policyRef)) {
				continue;
			}
		}

		if (CFDictionaryContainsKey(tSetting, _kSecTrustSettingsPolicyString)) {
			// Restricted to a hostname, not a root.
			continue;
		}

		CFNumberRef cfNum;
		if (CFDictionaryGetValueIfPresent(tSetting, _kSecTrustSettingsResult, (const void**)&cfNum)) {
			CFNumberGetValue(cfNum, kCFNumberSInt32Type, &result);
		} else {
			// > If this key is not present, a default value of
			// > kSecTrustSettingsResultTrustRoot is assumed.
			result = kSecTrustSettingsResultTrustRoot;
		}

		// If multiple dictionaries match, we are supposed to "OR" them,
		// the semantics of which are not clear. Since TrustRoot and TrustAsRoot
		// are mutually exclusive, Deny should probably override, and Invalid and
		// Unspecified be overridden, approximate this by stopping at the first
		// TrustRoot, TrustAsRoot or Deny.
		if (result == kSecTrustSettingsResultTrustRoot) {
			break;
		} else if (result == kSecTrustSettingsResultTrustAsRoot) {
			break;
		} else if (result == kSecTrustSettingsResultDeny) {
			break;
		}
	}

	// If trust settings are present, but none of them match the policy...
	// the docs don't tell us what to do.
	//
	// "Trust settings for a given use apply if any of the dictionaries in the
	// certificate’s trust settings array satisfies the specified use." suggests
	// that it's as if there were no trust settings at all, so we should probably
	// fallback to the admin trust settings. TODO.
	if (result == 0) {
		result = kSecTrustSettingsResultUnspecified;
	}

	CFRelease(_kSecTrustSettingsPolicy);
	CFRelease(_kSecTrustSettingsPolicyString);
	CFRelease(_kSecTrustSettingsResult);
	CFRelease(trustSettings);

	return result;
}

// isRootCertificate reports whether Subject and Issuer match.
static Boolean isRootCertificate(SecCertificateRef cert, CFErrorRef *errRef) {
	CFDataRef subjectName = SecCertificateCopyNormalizedSubjectContent(cert, errRef);
	if (*errRef != NULL) {
		return false;
	}
	CFDataRef issuerName = SecCertificateCopyNormalizedIssuerContent(cert, errRef);
	if (*errRef != NULL) {
		CFRelease(subjectName);
		return false;
	}
	Boolean equal = CFEqual(subjectName, issuerName);
	CFRelease(subjectName);
	CFRelease(issuerName);
	return equal;
}

// CopyPEMRoots fetches the system's list of trusted X.509 root certificates
// for the kSecTrustSettingsPolicy SSL.
//
// On success it returns 0 and fills pemRoots with a CFDataRef that contains the extracted root
// certificates of the system. On failure, the function returns -1.
// Additionally, it fills untrustedPemRoots with certs that must be removed from pemRoots.
//
// Note: The CFDataRef returned in pemRoots and untrustedPemRoots must
// be released (using CFRelease) after we've consumed its content.
int CopyPEMRoots(CFDataRef *pemRoots, CFDataRef *untrustedPemRoots, bool debugDarwinRoots) {
	int i;

	if (debugDarwinRoots) {
		fprintf(stderr, "crypto/x509: kSecTrustSettingsResultInvalid = %d\n", kSecTrustSettingsResultInvalid);
		fprintf(stderr, "crypto/x509: kSecTrustSettingsResultTrustRoot = %d\n", kSecTrustSettingsResultTrustRoot);
		fprintf(stderr, "crypto/x509: kSecTrustSettingsResultTrustAsRoot = %d\n", kSecTrustSettingsResultTrustAsRoot);
		fprintf(stderr, "crypto/x509: kSecTrustSettingsResultDeny = %d\n", kSecTrustSettingsResultDeny);
		fprintf(stderr, "crypto/x509: kSecTrustSettingsResultUnspecified = %d\n", kSecTrustSettingsResultUnspecified);
	}

	// Get certificates from all domains, not just System, this lets
	// the user add CAs to their "login" keychain, and Admins to add
	// to the "System" keychain
	SecTrustSettingsDomain domains[] = { kSecTrustSettingsDomainSystem,
		kSecTrustSettingsDomainAdmin, kSecTrustSettingsDomainUser };

	int numDomains = sizeof(domains)/sizeof(SecTrustSettingsDomain);
	if (pemRoots == NULL || untrustedPemRoots == NULL) {
		return -1;
	}

	CFMutableDataRef combinedData = CFDataCreateMutable(kCFAllocatorDefault, 0);
	CFMutableDataRef combinedUntrustedData = CFDataCreateMutable(kCFAllocatorDefault, 0);
	for (i = 0; i < numDomains; i++) {
		int j;
		CFArrayRef certs = NULL;
		OSStatus err = SecTrustSettingsCopyCertificates(domains[i], &certs);
		if (err != noErr) {
			continue;
		}

		CFIndex numCerts = CFArrayGetCount(certs);
		for (j = 0; j < numCerts; j++) {
			SecCertificateRef cert = (SecCertificateRef)CFArrayGetValueAtIndex(certs, j);
			if (cert == NULL) {
				continue;
			}

			SInt32 result;
			if (domains[i] == kSecTrustSettingsDomainSystem) {
				// Certs found in the system domain are always trusted. If the user
				// configures "Never Trust" on such a cert, it will also be found in the
				// admin or user domain, causing it to be added to untrustedPemRoots. The
				// Go code will then clean this up.
				result = kSecTrustSettingsResultTrustRoot;
			} else {
				result = sslTrustSettingsResult(cert);
				if (debugDarwinRoots) {
					CFErrorRef errRef = NULL;
					CFStringRef summary = SecCertificateCopyShortDescription(NULL, cert, &errRef);
					if (errRef != NULL) {
						fprintf(stderr, "crypto/x509: SecCertificateCopyShortDescription failed\n");
						CFRelease(errRef);
						continue;
					}

					CFIndex length = CFStringGetLength(summary);
					CFIndex maxSize = CFStringGetMaximumSizeForEncoding(length, kCFStringEncodingUTF8) + 1;
					char *buffer = malloc(maxSize);
					if (CFStringGetCString(summary, buffer, maxSize, kCFStringEncodingUTF8)) {
						fprintf(stderr, "crypto/x509: %s returned %d\n", buffer, (int)result);
					}
					free(buffer);
					CFRelease(summary);
				}
			}

			CFMutableDataRef appendTo;
			// > Note the distinction between the results kSecTrustSettingsResultTrustRoot
			// > and kSecTrustSettingsResultTrustAsRoot: The former can only be applied to
			// > root (self-signed) certificates; the latter can only be applied to
			// > non-root certificates.
			if (result == kSecTrustSettingsResultTrustRoot) {
				CFErrorRef errRef = NULL;
				if (!isRootCertificate(cert, &errRef) || errRef != NULL) {
					if (errRef != NULL) CFRelease(errRef);
					continue;
				}

				appendTo = combinedData;
			} else if (result == kSecTrustSettingsResultTrustAsRoot) {
				CFErrorRef errRef = NULL;
				if (isRootCertificate(cert, &errRef) || errRef != NULL) {
					if (errRef != NULL) CFRelease(errRef);
					continue;
				}

				appendTo = combinedData;
			} else if (result == kSecTrustSettingsResultDeny) {
				appendTo = combinedUntrustedData;
			} else if (result == kSecTrustSettingsResultUnspecified) {
				// Certificates with unspecified trust should probably be added to a pool of
				// intermediates for chain building, or checked for transitive trust and
				// added to the root pool (which is an imprecise approximation because it
				// cuts chains short) but we don't support either at the moment. TODO.
				continue;
			} else {
				continue;
			}

			CFDataRef data = NULL;
			err = SecItemExport(cert, kSecFormatX509Cert, kSecItemPemArmour, NULL, &data);
			if (err != noErr) {
				continue;
			}
			if (data != NULL) {
				CFDataAppendBytes(appendTo, CFDataGetBytePtr(data), CFDataGetLength(data));
				CFRelease(data);
			}
		}
		CFRelease(certs);
	}
	*pemRoots = combinedData;
	*untrustedPemRoots = combinedUntrustedData;
	return 0;
}
*/
import "C"
import (
	"errors"
	"unsafe"
)

func loadSystemRoots() (*CertPool, error) {
	var data, untrustedData C.CFDataRef
	err := C.CopyPEMRoots(&data, &untrustedData, C.bool(debugDarwinRoots))
	if err == -1 {
		return nil, errors.New("crypto/x509: failed to load darwin system roots with cgo")
	}
	defer C.CFRelease(C.CFTypeRef(data))
	defer C.CFRelease(C.CFTypeRef(untrustedData))

	buf := C.GoBytes(unsafe.Pointer(C.CFDataGetBytePtr(data)), C.int(C.CFDataGetLength(data)))
	roots := NewCertPool()
	roots.AppendCertsFromPEM(buf)

	if C.CFDataGetLength(untrustedData) == 0 {
		return roots, nil
	}

	buf = C.GoBytes(unsafe.Pointer(C.CFDataGetBytePtr(untrustedData)), C.int(C.CFDataGetLength(untrustedData)))
	untrustedRoots := NewCertPool()
	untrustedRoots.AppendCertsFromPEM(buf)

	trustedRoots := NewCertPool()
	for _, c := range roots.certs {
		if !untrustedRoots.contains(c) {
			trustedRoots.AddCert(c)
		}
	}
	return trustedRoots, nil
}
