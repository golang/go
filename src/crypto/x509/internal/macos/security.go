// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin

package macOS

import (
	"errors"
	"fmt"
	"internal/abi"
	"strconv"
	"unsafe"
)

// Security.framework linker flags for the external linker. See Issue 42459.
//
//go:cgo_ldflag "-framework"
//go:cgo_ldflag "Security"

// Based on https://opensource.apple.com/source/Security/Security-59306.41.2/base/Security.h

type SecTrustSettingsResult int32

const (
	SecTrustSettingsResultInvalid SecTrustSettingsResult = iota
	SecTrustSettingsResultTrustRoot
	SecTrustSettingsResultTrustAsRoot
	SecTrustSettingsResultDeny
	SecTrustSettingsResultUnspecified
)

type SecTrustResultType int32

const (
	SecTrustResultInvalid SecTrustResultType = iota
	SecTrustResultProceed
	SecTrustResultConfirm // deprecated
	SecTrustResultDeny
	SecTrustResultUnspecified
	SecTrustResultRecoverableTrustFailure
	SecTrustResultFatalTrustFailure
	SecTrustResultOtherError
)

type SecTrustSettingsDomain int32

const (
	SecTrustSettingsDomainUser SecTrustSettingsDomain = iota
	SecTrustSettingsDomainAdmin
	SecTrustSettingsDomainSystem
)

type OSStatus struct {
	call   string
	status int32
}

func (s OSStatus) Error() string {
	return s.call + " error: " + strconv.Itoa(int(s.status))
}

// Dictionary keys are defined as build-time strings with CFSTR, but the Go
// linker's internal linking mode can't handle CFSTR relocations. Create our
// own dynamic strings instead and just never release them.
//
// Note that this might be the only thing that can break over time if
// these values change, as the ABI arguably requires using the strings
// pointed to by the symbols, not values that happen to be equal to them.

var SecTrustSettingsResultKey = StringToCFString("kSecTrustSettingsResult")
var SecTrustSettingsPolicy = StringToCFString("kSecTrustSettingsPolicy")
var SecTrustSettingsPolicyString = StringToCFString("kSecTrustSettingsPolicyString")
var SecPolicyOid = StringToCFString("SecPolicyOid")
var SecPolicyAppleSSL = StringToCFString("1.2.840.113635.100.1.3") // defined by POLICYMACRO

var ErrNoTrustSettings = errors.New("no trust settings found")

const errSecNoTrustSettings = -25263

//go:cgo_import_dynamic x509_SecTrustSettingsCopyCertificates SecTrustSettingsCopyCertificates "/System/Library/Frameworks/Security.framework/Versions/A/Security"

func SecTrustSettingsCopyCertificates(domain SecTrustSettingsDomain) (certArray CFRef, err error) {
	ret := syscall(abi.FuncPCABI0(x509_SecTrustSettingsCopyCertificates_trampoline), uintptr(domain),
		uintptr(unsafe.Pointer(&certArray)), 0, 0, 0, 0)
	if int32(ret) == errSecNoTrustSettings {
		return 0, ErrNoTrustSettings
	} else if ret != 0 {
		return 0, OSStatus{"SecTrustSettingsCopyCertificates", int32(ret)}
	}
	return certArray, nil
}
func x509_SecTrustSettingsCopyCertificates_trampoline()

const errSecItemNotFound = -25300

//go:cgo_import_dynamic x509_SecTrustSettingsCopyTrustSettings SecTrustSettingsCopyTrustSettings "/System/Library/Frameworks/Security.framework/Versions/A/Security"

func SecTrustSettingsCopyTrustSettings(cert CFRef, domain SecTrustSettingsDomain) (trustSettings CFRef, err error) {
	ret := syscall(abi.FuncPCABI0(x509_SecTrustSettingsCopyTrustSettings_trampoline), uintptr(cert), uintptr(domain),
		uintptr(unsafe.Pointer(&trustSettings)), 0, 0, 0)
	if int32(ret) == errSecItemNotFound {
		return 0, ErrNoTrustSettings
	} else if ret != 0 {
		return 0, OSStatus{"SecTrustSettingsCopyTrustSettings", int32(ret)}
	}
	return trustSettings, nil
}
func x509_SecTrustSettingsCopyTrustSettings_trampoline()

//go:cgo_import_dynamic x509_SecPolicyCopyProperties SecPolicyCopyProperties "/System/Library/Frameworks/Security.framework/Versions/A/Security"

func SecPolicyCopyProperties(policy CFRef) CFRef {
	ret := syscall(abi.FuncPCABI0(x509_SecPolicyCopyProperties_trampoline), uintptr(policy), 0, 0, 0, 0, 0)
	return CFRef(ret)
}
func x509_SecPolicyCopyProperties_trampoline()

//go:cgo_import_dynamic x509_SecTrustCreateWithCertificates SecTrustCreateWithCertificates "/System/Library/Frameworks/Security.framework/Versions/A/Security"

func SecTrustCreateWithCertificates(certs CFRef, policies CFRef) (CFRef, error) {
	var trustObj CFRef
	ret := syscall(abi.FuncPCABI0(x509_SecTrustCreateWithCertificates_trampoline), uintptr(certs), uintptr(policies),
		uintptr(unsafe.Pointer(&trustObj)), 0, 0, 0)
	if int32(ret) != 0 {
		return 0, OSStatus{"SecTrustCreateWithCertificates", int32(ret)}
	}
	return trustObj, nil
}
func x509_SecTrustCreateWithCertificates_trampoline()

//go:cgo_import_dynamic x509_SecCertificateCreateWithData SecCertificateCreateWithData "/System/Library/Frameworks/Security.framework/Versions/A/Security"

func SecCertificateCreateWithData(b []byte) (CFRef, error) {
	data := BytesToCFData(b)
	defer CFRelease(data)
	ret := syscall(abi.FuncPCABI0(x509_SecCertificateCreateWithData_trampoline), kCFAllocatorDefault, uintptr(data), 0, 0, 0, 0)
	// Returns NULL if the data passed in the data parameter is not a valid
	// DER-encoded X.509 certificate.
	if ret == 0 {
		return 0, errors.New("SecCertificateCreateWithData: invalid certificate")
	}
	return CFRef(ret), nil
}
func x509_SecCertificateCreateWithData_trampoline()

//go:cgo_import_dynamic x509_SecPolicyCreateSSL SecPolicyCreateSSL "/System/Library/Frameworks/Security.framework/Versions/A/Security"

func SecPolicyCreateSSL(name string) CFRef {
	var hostname CFString
	if name != "" {
		hostname = StringToCFString(name)
		defer CFRelease(CFRef(hostname))
	}
	ret := syscall(abi.FuncPCABI0(x509_SecPolicyCreateSSL_trampoline), 1 /* true */, uintptr(hostname), 0, 0, 0, 0)
	return CFRef(ret)
}
func x509_SecPolicyCreateSSL_trampoline()

//go:cgo_import_dynamic x509_SecTrustSetVerifyDate SecTrustSetVerifyDate "/System/Library/Frameworks/Security.framework/Versions/A/Security"

func SecTrustSetVerifyDate(trustObj CFRef, dateRef CFRef) error {
	ret := syscall(abi.FuncPCABI0(x509_SecTrustSetVerifyDate_trampoline), uintptr(trustObj), uintptr(dateRef), 0, 0, 0, 0)
	if int32(ret) != 0 {
		return OSStatus{"SecTrustSetVerifyDate", int32(ret)}
	}
	return nil
}
func x509_SecTrustSetVerifyDate_trampoline()

//go:cgo_import_dynamic x509_SecTrustEvaluate SecTrustEvaluate "/System/Library/Frameworks/Security.framework/Versions/A/Security"

func SecTrustEvaluate(trustObj CFRef) (CFRef, error) {
	var result CFRef
	ret := syscall(abi.FuncPCABI0(x509_SecTrustEvaluate_trampoline), uintptr(trustObj), uintptr(unsafe.Pointer(&result)), 0, 0, 0, 0)
	if int32(ret) != 0 {
		return 0, OSStatus{"SecTrustEvaluate", int32(ret)}
	}
	return CFRef(result), nil
}
func x509_SecTrustEvaluate_trampoline()

//go:cgo_import_dynamic x509_SecTrustGetResult SecTrustGetResult "/System/Library/Frameworks/Security.framework/Versions/A/Security"

func SecTrustGetResult(trustObj CFRef, result CFRef) (CFRef, CFRef, error) {
	var chain, info CFRef
	ret := syscall(abi.FuncPCABI0(x509_SecTrustGetResult_trampoline), uintptr(trustObj), uintptr(unsafe.Pointer(&result)),
		uintptr(unsafe.Pointer(&chain)), uintptr(unsafe.Pointer(&info)), 0, 0)
	if int32(ret) != 0 {
		return 0, 0, OSStatus{"SecTrustGetResult", int32(ret)}
	}
	return chain, info, nil
}
func x509_SecTrustGetResult_trampoline()

//go:cgo_import_dynamic x509_SecTrustEvaluateWithError SecTrustEvaluateWithError "/System/Library/Frameworks/Security.framework/Versions/A/Security"

func SecTrustEvaluateWithError(trustObj CFRef) error {
	var errRef CFRef
	ret := syscall(abi.FuncPCABI0(x509_SecTrustEvaluateWithError_trampoline), uintptr(trustObj), uintptr(unsafe.Pointer(&errRef)), 0, 0, 0, 0)
	if int32(ret) != 1 {
		errStr := CFErrorCopyDescription(errRef)
		err := fmt.Errorf("x509: %s", CFStringToString(errStr))
		CFRelease(errRef)
		CFRelease(errStr)
		return err
	}
	return nil
}
func x509_SecTrustEvaluateWithError_trampoline()

//go:cgo_import_dynamic x509_SecTrustGetCertificateCount SecTrustGetCertificateCount "/System/Library/Frameworks/Security.framework/Versions/A/Security"

func SecTrustGetCertificateCount(trustObj CFRef) int {
	ret := syscall(abi.FuncPCABI0(x509_SecTrustGetCertificateCount_trampoline), uintptr(trustObj), 0, 0, 0, 0, 0)
	return int(ret)
}
func x509_SecTrustGetCertificateCount_trampoline()

//go:cgo_import_dynamic x509_SecTrustGetCertificateAtIndex SecTrustGetCertificateAtIndex "/System/Library/Frameworks/Security.framework/Versions/A/Security"

func SecTrustGetCertificateAtIndex(trustObj CFRef, i int) CFRef {
	ret := syscall(abi.FuncPCABI0(x509_SecTrustGetCertificateAtIndex_trampoline), uintptr(trustObj), uintptr(i), 0, 0, 0, 0)
	return CFRef(ret)
}
func x509_SecTrustGetCertificateAtIndex_trampoline()

//go:cgo_import_dynamic x509_SecCertificateCopyData SecCertificateCopyData "/System/Library/Frameworks/Security.framework/Versions/A/Security"

func SecCertificateCopyData(cert CFRef) ([]byte, error) {
	ret := syscall(abi.FuncPCABI0(x509_SecCertificateCopyData_trampoline), uintptr(cert), 0, 0, 0, 0, 0)
	if ret == 0 {
		return nil, errors.New("x509: invalid certificate object")
	}
	b := CFDataToSlice(CFRef(ret))
	CFRelease(CFRef(ret))
	return b, nil
}
func x509_SecCertificateCopyData_trampoline()
