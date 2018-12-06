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

static bool isSSLPolicy(SecPolicyRef policyRef) {
	if (!policyRef) {
		return false;
	}
	CFDictionaryRef properties = SecPolicyCopyProperties(policyRef);
	if (properties == NULL) {
		return false;
	}
	CFTypeRef value = NULL;
	if (CFDictionaryGetValueIfPresent(properties, kSecPolicyOid, (const void **)&value)) {
		CFRelease(properties);
		return CFEqual(value, kSecPolicyAppleSSL);
	}
	CFRelease(properties);
	return false;
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

		// First, check if this trust setting applies to our policy. We assume
		// only one will. The docs suggest that there might be multiple applying
		// but don't explain how to combine them.
		SecPolicyRef policyRef;
		if (CFDictionaryGetValueIfPresent(tSetting, _kSecTrustSettingsPolicy, (const void**)&policyRef)) {
			if (!isSSLPolicy(policyRef)) {
				continue;
			}
		} else {
			continue;
		}

		if (CFDictionaryContainsKey(tSetting, _kSecTrustSettingsPolicyString)) {
			// Restricted to a hostname, not a root.
			continue;
		}

		CFNumberRef cfNum;
		if (CFDictionaryGetValueIfPresent(tSetting, _kSecTrustSettingsResult, (const void**)&cfNum)) {
			CFNumberGetValue(cfNum, kCFNumberSInt32Type, &result);
		} else {
			// > If the value of the kSecTrustSettingsResult component is not
			// > kSecTrustSettingsResultUnspecified for a usage constraints dictionary that has
			// > no constraints, the default value kSecTrustSettingsResultTrustRoot is assumed.
			result = kSecTrustSettingsResultTrustRoot;
		}

		break;
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

// FetchPEMRoots fetches the system's list of trusted X.509 root certificates
// for the kSecTrustSettingsPolicy SSL.
//
// On success it returns 0 and fills pemRoots with a CFDataRef that contains the extracted root
// certificates of the system. On failure, the function returns -1.
// Additionally, it fills untrustedPemRoots with certs that must be removed from pemRoots.
//
// Note: The CFDataRef returned in pemRoots and untrustedPemRoots must
// be released (using CFRelease) after we've consumed its content.
int FetchPEMRoots(CFDataRef *pemRoots, CFDataRef *untrustedPemRoots, bool debugDarwinRoots) {
	int i;

	if (debugDarwinRoots) {
		printf("crypto/x509: kSecTrustSettingsResultInvalid = %d\n", kSecTrustSettingsResultInvalid);
		printf("crypto/x509: kSecTrustSettingsResultTrustRoot = %d\n", kSecTrustSettingsResultTrustRoot);
		printf("crypto/x509: kSecTrustSettingsResultTrustAsRoot = %d\n", kSecTrustSettingsResultTrustAsRoot);
		printf("crypto/x509: kSecTrustSettingsResultDeny = %d\n", kSecTrustSettingsResultDeny);
		printf("crypto/x509: kSecTrustSettingsResultUnspecified = %d\n", kSecTrustSettingsResultUnspecified);
	}

	// Get certificates from all domains, not just System, this lets
	// the user add CAs to their "login" keychain, and Admins to add
	// to the "System" keychain
	SecTrustSettingsDomain domains[] = { kSecTrustSettingsDomainSystem,
		kSecTrustSettingsDomainAdmin, kSecTrustSettingsDomainUser };

	int numDomains = sizeof(domains)/sizeof(SecTrustSettingsDomain);
	if (pemRoots == NULL) {
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
			CFDataRef data = NULL;
			CFArrayRef trustSettings = NULL;
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
						printf("crypto/x509: SecCertificateCopyShortDescription failed\n");
						CFRelease(errRef);
						continue;
					}

					CFIndex length = CFStringGetLength(summary);
					CFIndex maxSize = CFStringGetMaximumSizeForEncoding(length, kCFStringEncodingUTF8) + 1;
					char *buffer = malloc(maxSize);
					if (CFStringGetCString(summary, buffer, maxSize, kCFStringEncodingUTF8)) {
						printf("crypto/x509: %s returned %d\n", buffer, (int)result);
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
				continue;
			} else {
				continue;
			}

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
	roots := NewCertPool()

	var data C.CFDataRef = 0
	var untrustedData C.CFDataRef = 0
	err := C.FetchPEMRoots(&data, &untrustedData, C.bool(debugDarwinRoots))
	if err == -1 {
		return nil, errors.New("crypto/x509: failed to load darwin system roots with cgo")
	}

	defer C.CFRelease(C.CFTypeRef(data))
	buf := C.GoBytes(unsafe.Pointer(C.CFDataGetBytePtr(data)), C.int(C.CFDataGetLength(data)))
	roots.AppendCertsFromPEM(buf)
	if untrustedData == 0 {
		return roots, nil
	}
	defer C.CFRelease(C.CFTypeRef(untrustedData))
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
