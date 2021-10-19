// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sym

import (
	"cmd/internal/obj"
	"internal/buildcfg"
)

const (
	SymVerABI0        = 0
	SymVerABIInternal = 1
	SymVerABICount    = 2  // Number of internal ABIs
	SymVerStatic      = 10 // Minimum version used by static (file-local) syms
)

func ABIToVersion(abi obj.ABI) int {
	switch abi {
	case obj.ABI0:
		return SymVerABI0
	case obj.ABIInternal:
		if !buildcfg.Experiment.RegabiWrappers {
			// If wrappers are not enabled, ABI0 and ABIInternal are actually same
			// so we normalize everything to ABI0.
			return SymVerABI0
		}
		return SymVerABIInternal
	}
	return -1
}

func VersionToABI(v int) (obj.ABI, bool) {
	switch v {
	case SymVerABI0:
		return obj.ABI0, true
	case SymVerABIInternal:
		return obj.ABIInternal, true
	}
	return ^obj.ABI(0), false
}
