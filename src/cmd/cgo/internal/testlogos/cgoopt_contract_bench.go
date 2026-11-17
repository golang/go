// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlogos

/*
#include <stdint.h>
uint32_t logos_contract_add(uint32_t, uint32_t);
#define LOGOS_CONTRACT_LOCAL 11
static uint32_t logos_contract_first(void) { return LOGOS_CONTRACT_LOCAL; }
*/
import "C"

func callCgoOptLateAdd(a, b uint32) uint32 {
	return uint32(C.logos_contract_add(C.uint32_t(a), C.uint32_t(b)))
}

func callCgoOptLocalFirst() uint32 { return uint32(C.logos_contract_first()) }
