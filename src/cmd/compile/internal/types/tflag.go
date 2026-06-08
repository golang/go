// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"cmd/compile/internal/base"
	"internal/abi"
	"strings"
	"sync"
)

// tflagComputed is set in the high bit of Type.tflag once the low
// bits hold a valid abi.TFlag. abi.TFlag itself uses only the low 6.
const tflagComputed uint8 = 1 << 7

var tflagMu sync.Mutex

// TFlag returns the abi.TFlag value for t's runtime type. Callers
// must have run typecheck.CalcMethods on ReceiverBaseType(t).
func (t *Type) TFlag() abi.TFlag {
	tflagMu.Lock()
	defer tflagMu.Unlock()
	if t.tflag&tflagComputed != 0 {
		return abi.TFlag(t.tflag &^ tflagComputed)
	}
	tflag := computeTFlag(t)
	t.tflag = uint8(tflag) | tflagComputed
	return tflag
}

func computeTFlag(t *Type) abi.TFlag {
	var tflag abi.TFlag
	if hasUncommon(t) {
		tflag |= abi.TFlagUncommon
	}
	if t.Sym() != nil && t.Sym().Name != "" {
		tflag |= abi.TFlagNamed
	}
	if t.alg == AMEM {
		tflag |= abi.TFlagRegularMemory
	}
	if PtrDataSize(t)/int64(PtrSize) > abi.MaxPtrmaskBytes*8 {
		tflag |= abi.TFlagGCMaskOnDemand
	}
	if !strings.HasPrefix(t.NameString(), "*") {
		tflag |= abi.TFlagExtraStar
	}
	if IsDirectIface(t) {
		tflag |= abi.TFlagDirectIface
	}
	return tflag
}

// hasUncommon reports whether t needs an abi.UncommonType.
// See TFlag for the precondition.
func hasUncommon(t *Type) bool {
	if t.Sym() != nil {
		return true
	}
	if t.HasShape() {
		return false
	}
	mt := ReceiverBaseType(t)
	if mt == nil {
		return false
	}
	if !mt.MethodsComputed() {
		base.Fatalf("hasUncommon: methods not computed on %v", mt)
	}
	for _, f := range mt.AllMethods() {
		if f.Nointerface() {
			continue
		}
		if !IsMethodApplicable(t, f) {
			continue
		}
		return true
	}
	return false
}
