// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build cgo,!arm,!arm64,!ios

package x509

/*
#cgo CFLAGS: -mmacosx-version-min=10.6 -D__MAC_OS_X_VERSION_MAX_ALLOWED=1060
#cgo LDFLAGS: -framework CoreFoundation -framework Security

#include <CoreFoundation/CoreFoundation.h>
#include <Security/Security.h>

// FetchPEMRoots fetches the system's list of trusted X.509 root certificates.
//
// On success it returns 0 and fills pemRoots with a CFDataRef that contains the extracted root
// certificates of the system. On failure, the function returns -1.
//
// Note: The CFDataRef returned in pemRoots must be released (using CFRelease) after
// we've consumed its content.
int FetchPEMRoots(CFDataRef *pemRoots) {
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

	CFMutableDataRef combinedData = CFDataCreateMutable(kCFAllocatorDefault, 0);
	for (int i = 0; i < numDomains; i++) {
		CFArrayRef certs = NULL;
		// Only get certificates from domain that are trusted
		OSStatus err = SecTrustSettingsCopyCertificates(domains[i], &certs);
		if (err != noErr) {
			continue;
		}

		int numCerts = CFArrayGetCount(certs);
		for (int j = 0; j < numCerts; j++) {
			CFDataRef data = NULL;
			CFErrorRef errRef = NULL;
			SecCertificateRef cert = (SecCertificateRef)CFArrayGetValueAtIndex(certs, j);
			if (cert == NULL) {
				continue;
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
				CFDataAppendBytes(combinedData, CFDataGetBytePtr(data), CFDataGetLength(data));
				CFRelease(data);
			}
		}
		CFRelease(certs);
	}
	*pemRoots = combinedData;
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
	err := C.FetchPEMRoots(&data)
	if err == -1 {
		// TODO: better error message
		return nil, errors.New("crypto/x509: failed to load darwin system roots with cgo")
	}

	defer C.CFRelease(C.CFTypeRef(data))
	buf := C.GoBytes(unsafe.Pointer(C.CFDataGetBytePtr(data)), C.int(C.CFDataGetLength(data)))
	roots.AppendCertsFromPEM(buf)
	return roots, nil
}
