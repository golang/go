// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin && !ios
// +build darwin,!ios

#include "textflag.h"

// The trampolines are ABIInternal as they are address-taken in
// Go code.

TEXT ·x509_SecTrustSettingsCopyCertificates_trampoline(SB),NOSPLIT,$0-0
	JMP	x509_SecTrustSettingsCopyCertificates(SB)
TEXT ·x509_SecItemExport_trampoline(SB),NOSPLIT,$0-0
	JMP	x509_SecItemExport(SB)
TEXT ·x509_SecTrustSettingsCopyTrustSettings_trampoline(SB),NOSPLIT,$0-0
	JMP	x509_SecTrustSettingsCopyTrustSettings(SB)
TEXT ·x509_SecPolicyCopyProperties_trampoline(SB),NOSPLIT,$0-0
	JMP	x509_SecPolicyCopyProperties(SB)
