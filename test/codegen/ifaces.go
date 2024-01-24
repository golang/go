// asmcheck

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

type I interface{ M() }

func NopConvertIface(x I) I {
	// amd64:-`.*runtime.convI2I`
	return I(x)
}

func NopConvertGeneric[T any](x T) T {
	// amd64:-`.*runtime.convI2I`
	return T(x)
}

var NopConvertGenericIface = NopConvertGeneric[I]

func ConvToM(x any) I {
	// amd64:`CALL\truntime.typeAssert`,`MOVL\t16\(.*\)`,`MOVQ\t8\(.*\)(.*\*1)`
	// arm64:`CALL\truntime.typeAssert`,`LDAR`,`MOVWU`,`MOVD\t\(R.*\)\(R.*\)`
	return x.(I)
}
