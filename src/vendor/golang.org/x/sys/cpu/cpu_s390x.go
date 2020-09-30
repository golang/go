// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cpu

const cacheLineSize = 256

func initOptions() {
	options = []option{
		{Name: "zarch", Feature: &S390X.HasZARCH},
		{Name: "stfle", Feature: &S390X.HasSTFLE},
		{Name: "ldisp", Feature: &S390X.HasLDISP},
		{Name: "eimm", Feature: &S390X.HasEIMM},
		{Name: "dfp", Feature: &S390X.HasDFP},
		{Name: "etf3eh", Feature: &S390X.HasETF3EH},
		{Name: "msa", Feature: &S390X.HasMSA},
		{Name: "aes", Feature: &S390X.HasAES},
		{Name: "aescbc", Feature: &S390X.HasAESCBC},
		{Name: "aesctr", Feature: &S390X.HasAESCTR},
		{Name: "aesgcm", Feature: &S390X.HasAESGCM},
		{Name: "ghash", Feature: &S390X.HasGHASH},
		{Name: "sha1", Feature: &S390X.HasSHA1},
		{Name: "sha256", Feature: &S390X.HasSHA256},
		{Name: "sha3", Feature: &S390X.HasSHA3},
		{Name: "sha512", Feature: &S390X.HasSHA512},
		{Name: "vx", Feature: &S390X.HasVX},
		{Name: "vxe", Feature: &S390X.HasVXE},
	}
}
