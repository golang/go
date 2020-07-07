// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin,amd64

#include "textflag.h"

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
