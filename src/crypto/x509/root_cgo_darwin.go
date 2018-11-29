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

// FetchPEMRoots fetches the system's list of trusted X.509 root certificates.
//
// On success it returns 0 and fills pemRoots with a CFDataRef that contains the extracted root
// certificates of the system. On failure, the function returns -1.
// Additionally, it fills untrustedPemRoots with certs that must be removed from pemRoots.
//
// Note: The CFDataRef returned in pemRoots and untrustedPemRoots must
// be released (using CFRelease) after we've consumed its content.
int FetchPEMRoots(CFDataRef *pemRoots, CFDataRef *untrustedPemRoots) {
	int i;

	// Get certificates from all domains, not just System, this lets
	// the user add CAs to their "login" keychain, and Admins to add
	// to the "System" keychain
	SecTrustSettingsDomain domains[] = { kSecTrustSettingsDomainSystem,
					     kSecTrustSettingsDomainAdmin,
					     kSecTrustSettingsDomainUser };

	int numDomains = sizeof(domains)/sizeof(SecTrustSettingsDomain);
	if (pemRoots == NULL) {
		return -1;
	}

	// kSecTrustSettingsResult is defined as CFSTR("kSecTrustSettingsResult"),
	// but the Go linker's internal linking mode can't handle CFSTR relocations.
	// Create our own dynamic string instead and release it below.
	CFStringRef policy = CFStringCreateWithCString(NULL, "kSecTrustSettingsResult", kCFStringEncodingUTF8);

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
			CFErrorRef errRef = NULL;
			CFArrayRef trustSettings = NULL;
			SecCertificateRef cert = (SecCertificateRef)CFArrayGetValueAtIndex(certs, j);
			if (cert == NULL) {
				continue;
			}
			// We only want trusted certs.
			int untrusted = 0;
			int trustAsRoot = 0;
			int trustRoot = 0;
			if (i == 0) {
				trustAsRoot = 1;
			} else {
				int k;
				CFIndex m;

				// Certs found in the system domain are always trusted. If the user
				// configures "Never Trust" on such a cert, it will also be found in the
				// admin or user domain, causing it to be added to untrustedPemRoots. The
				// Go code will then clean this up.

				// Trust may be stored in any of the domains. According to Apple's
				// SecTrustServer.c, "user trust settings overrule admin trust settings",
				// so take the last trust settings array we find.
				// Skip the system domain since it is always trusted.
				for (k = i; k < numDomains; k++) {
					CFArrayRef domainTrustSettings = NULL;
					err = SecTrustSettingsCopyTrustSettings(cert, domains[k], &domainTrustSettings);
					if (err == errSecSuccess && domainTrustSettings != NULL) {
						if (trustSettings) {
							CFRelease(trustSettings);
						}
						trustSettings = domainTrustSettings;
					}
				}
				if (trustSettings == NULL) {
					// "this certificate must be verified to a known trusted certificate"; aka not a root.
					continue;
				}
				for (m = 0; m < CFArrayGetCount(trustSettings); m++) {
					CFNumberRef cfNum;
					CFDictionaryRef tSetting = (CFDictionaryRef)CFArrayGetValueAtIndex(trustSettings, m);
					if (CFDictionaryGetValueIfPresent(tSetting, policy, (const void**)&cfNum)){
						SInt32 result = 0;
						CFNumberGetValue(cfNum, kCFNumberSInt32Type, &result);
						// TODO: The rest of the dictionary specifies conditions for evaluation.
						if (result == kSecTrustSettingsResultDeny) {
							untrusted = 1;
						} else if (result == kSecTrustSettingsResultTrustAsRoot) {
							trustAsRoot = 1;
						} else if (result == kSecTrustSettingsResultTrustRoot) {
							trustRoot = 1;
						}
					}
				}
				CFRelease(trustSettings);
			}

			if (trustRoot) {
				// We only want to add Root CAs, so make sure Subject and Issuer Name match
				CFDataRef subjectName = SecCertificateCopyNormalizedSubjectContent(cert, &errRef);
				if (errRef != NULL) {
					CFRelease(errRef);
					continue;
				}
				CFDataRef issuerName = SecCertificateCopyNormalizedIssuerContent(cert, &errRef);
				if (errRef != NULL) {
					CFRelease(subjectName);
					CFRelease(errRef);
					continue;
				}
				Boolean equal = CFEqual(subjectName, issuerName);
				CFRelease(subjectName);
				CFRelease(issuerName);
				if (!equal) {
					continue;
				}
			}

			err = SecItemExport(cert, kSecFormatX509Cert, kSecItemPemArmour, NULL, &data);
			if (err != noErr) {
				continue;
			}

			if (data != NULL) {
				if (!trustRoot && !trustAsRoot) {
					untrusted = 1;
				}
				CFMutableDataRef appendTo = untrusted ? combinedUntrustedData : combinedData;
				CFDataAppendBytes(appendTo, CFDataGetBytePtr(data), CFDataGetLength(data));
				CFRelease(data);
			}
		}
		CFRelease(certs);
	}
	CFRelease(policy);
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
	err := C.FetchPEMRoots(&data, &untrustedData)
	if err == -1 {
		// TODO: better error message
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
