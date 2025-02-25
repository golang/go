// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin

#include "textflag.h"

// The trampolines are ABIInternal as they are address-taken in
// Go code.

TEXT ·x509_SecTrustCreateWithCertificates_trampoline(SB),NOSPLIT,$0-0
	JMP x509_SecTrustCreateWithCertificates(SB)
TEXT ·x509_SecCertificateCreateWithData_trampoline(SB),NOSPLIT,$0-0
	JMP x509_SecCertificateCreateWithData(SB)
TEXT ·x509_SecPolicyCreateSSL_trampoline(SB),NOSPLIT,$0-0
	JMP x509_SecPolicyCreateSSL(SB)
TEXT ·x509_SecTrustSetVerifyDate_trampoline(SB),NOSPLIT,$0-0
	JMP x509_SecTrustSetVerifyDate(SB)
TEXT ·x509_SecTrustEvaluate_trampoline(SB),NOSPLIT,$0-0
	JMP x509_SecTrustEvaluate(SB)
TEXT ·x509_SecTrustEvaluateWithError_trampoline(SB),NOSPLIT,$0-0
	JMP x509_SecTrustEvaluateWithError(SB)
TEXT ·x509_SecTrustGetCertificateCount_trampoline(SB),NOSPLIT,$0-0
	JMP x509_SecTrustGetCertificateCount(SB)
TEXT ·x509_SecTrustGetCertificateAtIndex_trampoline(SB),NOSPLIT,$0-0
	JMP x509_SecTrustGetCertificateAtIndex(SB)
TEXT ·x509_SecCertificateCopyData_trampoline(SB),NOSPLIT,$0-0
	JMP x509_SecCertificateCopyData(SB)
