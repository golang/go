// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin

#include "textflag.h"

// The trampolines are ABIInternal as they are address-taken in
// Go code.

TEXT ·x509_CFArrayGetCount_trampoline(SB),NOSPLIT,$0-0
	JMP	x509_CFArrayGetCount(SB)
TEXT ·x509_CFArrayGetValueAtIndex_trampoline(SB),NOSPLIT,$0-0
	JMP	x509_CFArrayGetValueAtIndex(SB)
TEXT ·x509_CFDataGetBytePtr_trampoline(SB),NOSPLIT,$0-0
	JMP	x509_CFDataGetBytePtr(SB)
TEXT ·x509_CFDataGetLength_trampoline(SB),NOSPLIT,$0-0
	JMP	x509_CFDataGetLength(SB)
TEXT ·x509_CFStringCreateWithBytes_trampoline(SB),NOSPLIT,$0-0
	JMP	x509_CFStringCreateWithBytes(SB)
TEXT ·x509_CFRelease_trampoline(SB),NOSPLIT,$0-0
	JMP	x509_CFRelease(SB)
TEXT ·x509_CFDictionaryGetValueIfPresent_trampoline(SB),NOSPLIT,$0-0
	JMP	x509_CFDictionaryGetValueIfPresent(SB)
TEXT ·x509_CFNumberGetValue_trampoline(SB),NOSPLIT,$0-0
	JMP	x509_CFNumberGetValue(SB)
TEXT ·x509_CFEqual_trampoline(SB),NOSPLIT,$0-0
	JMP	x509_CFEqual(SB)
TEXT ·x509_CFArrayCreateMutable_trampoline(SB),NOSPLIT,$0-0
	JMP x509_CFArrayCreateMutable(SB)
TEXT ·x509_CFArrayAppendValue_trampoline(SB),NOSPLIT,$0-0
	JMP x509_CFArrayAppendValue(SB)
TEXT ·x509_CFDateCreate_trampoline(SB),NOSPLIT,$0-0
	JMP x509_CFDateCreate(SB)
TEXT ·x509_CFDataCreate_trampoline(SB),NOSPLIT,$0-0
	JMP x509_CFDataCreate(SB)
TEXT ·x509_CFErrorCopyDescription_trampoline(SB),NOSPLIT,$0-0
	JMP x509_CFErrorCopyDescription(SB)
TEXT ·x509_CFErrorGetCode_trampoline(SB),NOSPLIT,$0-0
	JMP x509_CFErrorGetCode(SB)
TEXT ·x509_CFStringCreateExternalRepresentation_trampoline(SB),NOSPLIT,$0-0
	JMP x509_CFStringCreateExternalRepresentation(SB)
