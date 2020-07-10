// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin,amd64

package macOS

import (
	"errors"
	"strconv"
	"unsafe"
)

// Based on https://opensource.apple.com/source/Security/Security-59306.41.2/base/Security.h

type SecTrustSettingsResult int32

const (
	SecTrustSettingsResultInvalid SecTrustSettingsResult = iota
	SecTrustSettingsResultTrustRoot
	SecTrustSettingsResultTrustAsRoot
	SecTrustSettingsResultDeny
	SecTrustSettingsResultUnspecified
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

//go:linkname x509_SecTrustSettingsCopyCertificates x509_SecTrustSettingsCopyCertificates
//go:cgo_import_dynamic x509_SecTrustSettingsCopyCertificates SecTrustSettingsCopyCertificates "/System/Library/Frameworks/Security.framework/Versions/A/Security"

func SecTrustSettingsCopyCertificates(domain SecTrustSettingsDomain) (certArray CFRef, err error) {
	ret := syscall(funcPC(x509_SecTrustSettingsCopyCertificates_trampoline), uintptr(domain),
		uintptr(unsafe.Pointer(&certArray)), 0, 0, 0, 0)
	if int32(ret) == errSecNoTrustSettings {
		return 0, ErrNoTrustSettings
	} else if ret != 0 {
		return 0, OSStatus{"SecTrustSettingsCopyCertificates", int32(ret)}
	}
	return certArray, nil
}
func x509_SecTrustSettingsCopyCertificates_trampoline()

const kSecFormatX509Cert int32 = 9

//go:linkname x509_SecItemExport x509_SecItemExport
//go:cgo_import_dynamic x509_SecItemExport SecItemExport "/System/Library/Frameworks/Security.framework/Versions/A/Security"

func SecItemExport(cert CFRef) (data CFRef, err error) {
	ret := syscall(funcPC(x509_SecItemExport_trampoline), uintptr(cert), uintptr(kSecFormatX509Cert),
		0 /* flags */, 0 /* keyParams */, uintptr(unsafe.Pointer(&data)), 0)
	if ret != 0 {
		return 0, OSStatus{"SecItemExport", int32(ret)}
	}
	return data, nil
}
func x509_SecItemExport_trampoline()

const errSecItemNotFound = -25300

//go:linkname x509_SecTrustSettingsCopyTrustSettings x509_SecTrustSettingsCopyTrustSettings
//go:cgo_import_dynamic x509_SecTrustSettingsCopyTrustSettings SecTrustSettingsCopyTrustSettings "/System/Library/Frameworks/Security.framework/Versions/A/Security"

func SecTrustSettingsCopyTrustSettings(cert CFRef, domain SecTrustSettingsDomain) (trustSettings CFRef, err error) {
	ret := syscall(funcPC(x509_SecTrustSettingsCopyTrustSettings_trampoline), uintptr(cert), uintptr(domain),
		uintptr(unsafe.Pointer(&trustSettings)), 0, 0, 0)
	if int32(ret) == errSecItemNotFound {
		return 0, ErrNoTrustSettings
	} else if ret != 0 {
		return 0, OSStatus{"SecTrustSettingsCopyTrustSettings", int32(ret)}
	}
	return trustSettings, nil
}
func x509_SecTrustSettingsCopyTrustSettings_trampoline()

//go:linkname x509_SecPolicyCopyProperties x509_SecPolicyCopyProperties
//go:cgo_import_dynamic x509_SecPolicyCopyProperties SecPolicyCopyProperties "/System/Library/Frameworks/Security.framework/Versions/A/Security"

func SecPolicyCopyProperties(policy CFRef) CFRef {
	ret := syscall(funcPC(x509_SecPolicyCopyProperties_trampoline), uintptr(policy), 0, 0, 0, 0, 0)
	return CFRef(ret)
}
func x509_SecPolicyCopyProperties_trampoline()
