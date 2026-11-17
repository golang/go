// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlogos

/*
#include <stdint.h>

#cgo nocallback logos_contract_add
uint32_t logos_contract_add(uint32_t a, uint32_t b) { return a + b; }
#define LOGOS_CONTRACT_LOCAL 23
static uint32_t logos_contract_second(void) { return LOGOS_CONTRACT_LOCAL; }
*/
import "C"

func callCgoOptLocalSecond() uint32 { return uint32(C.logos_contract_second()) }
