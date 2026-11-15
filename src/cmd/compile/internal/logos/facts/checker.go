// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package facts

import (
	"slices"

	"cmd/compile/internal/types2"
)

func init() {
	types2.LogosImportedConstFuncSource = constFuncSource
	types2.LogosImportedParamProvenZero = paramProvenZero
	types2.LogosImportedConstMapLen = constMapLen
	types2.LogosImportedConstComposite = constComposite
}

func constMapLen(path, object string) (int, bool) {
	f, ok := Lookup(path)
	if !ok || f == nil {
		return 0, false
	}
	for i := range f.ConstMaps {
		if f.ConstMaps[i].Object == object && f.ConstMaps[i].Path == "" {
			return len(f.ConstMaps[i].Keys), true
		}
	}
	return 0, false
}

func constComposite(path, object string) bool {
	f, ok := Lookup(path)
	return ok && f != nil && slices.Contains(f.ConstComps, object)
}

func constFuncSource(path, object string) []byte {
	f, ok := Lookup(path)
	if !ok || f == nil {
		return nil
	}
	for _, c := range f.ConstFuncs {
		if c.Object == object {
			return c.Source
		}
	}
	return nil
}

func paramProvenZero(path, object string, param int) bool {
	if f, ok := Lookup(path); ok && f != nil && f.GoPackage {
		for i := range f.Fns {
			if f.Fns[i].Object != object {
				continue
			}
			if f.Fns[i].MayYieldOnly() {
				continue
			}
			return maskProvenZero(f.Fns[i].ParamMutated, param)
		}
	}
	return stdlibParamProvenZero(path, object, param)
}

func maskProvenZero(mask string, param int) bool {
	if mask == "" {
		return false
	}
	bits, ok := parseHexMask(mask)
	if !ok {
		return false
	}
	return !maskBit(bits, param)
}

var logosStdlibParamMasks = map[string]string{
	"bytes\x00Equal": "0",
}

func stdlibParamProvenZero(path, object string, param int) bool {
	mask, ok := logosStdlibParamMasks[path+"\x00"+object]
	if !ok {
		return false
	}
	return maskProvenZero(mask, param)
}
