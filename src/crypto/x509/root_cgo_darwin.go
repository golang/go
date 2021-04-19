// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build cgo,!arm,!arm64,!ios

package x509

/*
#cgo CFLAGS: -mmacosx-version-min=10.6 -D__MAC_OS_X_VERSION_MAX_ALLOWED=1080
#cgo LDFLAGS: -framework CoreFoundation -framework Security

#include <errno.h>
#include <sys/sysctl.h>

#include <CoreFoundation/CoreFoundation.h>
#include <Security/Security.h>

// FetchPEMRoots_MountainLion is the version of FetchPEMRoots from Go 1.6
// which still works on OS X 10.8 (Mountain Lion).
// It lacks support for admin & user cert domains.
// See golang.org/issue/16473
int FetchPEMRoots_MountainLion(CFDataRef *pemRoots) {
	if (pemRoots == NULL) {
		return -1;
	}
	CFArrayRef certs = NULL;
	OSStatus err = SecTrustCopyAnchorCertificates(&certs);
	if (err != noErr) {
		return -1;
	}
	CFMutableDataRef combinedData = CFDataCreateMutable(kCFAllocatorDefault, 0);
	int i, ncerts = CFArrayGetCount(certs);
	for (i = 0; i < ncerts; i++) {
		CFDataRef data = NULL;
		SecCertificateRef cert = (SecCertificateRef)CFArrayGetValueAtIndex(certs, i);
		if (cert == NULL) {
			continue;
		}
		// Note: SecKeychainItemExport is deprecated as of 10.7 in favor of SecItemExport.
		// Once we support weak imports via cgo we should prefer that, and fall back to this
		// for older systems.
		err = SecKeychainItemExport(cert, kSecFormatX509Cert, kSecItemPemArmour, NULL, &data);
		if (err != noErr) {
			continue;
		}
		if (data != NULL) {
			CFDataAppendBytes(combinedData, CFDataGetBytePtr(data), CFDataGetLength(data));
			CFRelease(data);
		}
	}
	CFRelease(certs);
	*pemRoots = combinedData;
	return 0;
}

// useOldCode reports whether the running machine is OS X 10.8 Mountain Lion
// or older. We only support Mountain Lion and higher, but we'll at least try our
// best on older machines and continue to use the old code path.
//
// See golang.org/issue/16473
int useOldCode() {
	char str[256];
	size_t size = sizeof(str);
	memset(str, 0, size);
	sysctlbyname("kern.osrelease", str, &size, NULL, 0);
	// OS X 10.8 is osrelease "12.*", 10.7 is 11.*, 10.6 is 10.*.
	// We never supported things before that.
	return memcmp(str, "12.", 3) == 0 || memcmp(str, "11.", 3) == 0 || memcmp(str, "10.", 3) == 0;
}

// FetchPEMRoots fetches the system's list of trusted X.509 root certificates.
//
// On success it returns 0 and fills pemRoots with a CFDataRef that contains the extracted root
// certificates of the system. On failure, the function returns -1.
// Additionally, it fills untrustedPemRoots with certs that must be removed from pemRoots.
//
// Note: The CFDataRef returned in pemRoots and untrustedPemRoots must
// be released (using CFRelease) after we've consumed its content.
int FetchPEMRoots(CFDataRef *pemRoots, CFDataRef *untrustedPemRoots) {
	if (useOldCode()) {
		return FetchPEMRoots_MountainLion(pemRoots);
	}

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
	for (int i = 0; i < numDomains; i++) {
		CFArrayRef certs = NULL;
		OSStatus err = SecTrustSettingsCopyCertificates(domains[i], &certs);
		if (err != noErr) {
			continue;
		}

		CFIndex numCerts = CFArrayGetCount(certs);
		for (int j = 0; j < numCerts; j++) {
			CFDataRef data = NULL;
			CFErrorRef errRef = NULL;
			CFArrayRef trustSettings = NULL;
			SecCertificateRef cert = (SecCertificateRef)CFArrayGetValueAtIndex(certs, j);
			if (cert == NULL) {
				continue;
			}
			// We only want trusted certs.
			int untrusted = 0;
			if (i != 0) {
				// Certs found in the system domain are always trusted. If the user
				// configures "Never Trust" on such a cert, it will also be found in the
				// admin or user domain, causing it to be added to untrustedPemRoots. The
				// Go code will then clean this up.

				// Trust may be stored in any of the domains. According to Apple's
				// SecTrustServer.c, "user trust settings overrule admin trust settings",
				// so take the last trust settings array we find.
				// Skip the system domain since it is always trusted.
				for (int k = 1; k < numDomains; k++) {
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
				for (CFIndex k = 0; k < CFArrayGetCount(trustSettings); k++) {
					CFNumberRef cfNum;
					CFDictionaryRef tSetting = (CFDictionaryRef)CFArrayGetValueAtIndex(trustSettings, k);
					if (CFDictionaryGetValueIfPresent(tSetting, policy, (const void**)&cfNum)){
						SInt32 result = 0;
						CFNumberGetValue(cfNum, kCFNumberSInt32Type, &result);
						// TODO: The rest of the dictionary specifies conditions for evaluation.
						if (result == kSecTrustSettingsResultDeny) {
							untrusted = 1;
						}
					}
				}
				CFRelease(trustSettings);
			}
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

			// Note: SecKeychainItemExport is deprecated as of 10.7 in favor of SecItemExport.
			// Once we support weak imports via cgo we should prefer that, and fall back to this
			// for older systems.
			err = SecKeychainItemExport(cert, kSecFormatX509Cert, kSecItemPemArmour, NULL, &data);
			if (err != noErr) {
				continue;
			}

			if (data != NULL) {
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

	var data C.CFDataRef = nil
	var untrustedData C.CFDataRef = nil
	err := C.FetchPEMRoots(&data, &untrustedData)
	if err == -1 {
		// TODO: better error message
		return nil, errors.New("crypto/x509: failed to load darwin system roots with cgo")
	}

	defer C.CFRelease(C.CFTypeRef(data))
	buf := C.GoBytes(unsafe.Pointer(C.CFDataGetBytePtr(data)), C.int(C.CFDataGetLength(data)))
	roots.AppendCertsFromPEM(buf)
	if untrustedData == nil {
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
