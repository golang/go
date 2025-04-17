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

func e1(x any, y *int) bool {
	// amd64:-`.*faceeq`,`SETEQ`
	// arm64:-`.*faceeq`,`CSET\tEQ`
	return x == y
}

func e2(x any, y *int) bool {
	// amd64:-`.*faceeq`,`SETEQ`
	// arm64:-`.*faceeq`,`CSET\tEQ`
	return y == x
}

type E *int

func e3(x any, y E) bool {
	// amd64:-`.*faceeq`,`SETEQ`
	// arm64:-`.*faceeq`,`CSET\tEQ`
	return x == y
}

type T int

func (t *T) M() {}

func i1(x I, y *T) bool {
	// amd64:-`.*faceeq`,`SETEQ`
	// arm64:-`.*faceeq`,`CSET\tEQ`
	return x == y
}

func i2(x I, y *T) bool {
	// amd64:-`.*faceeq`,`SETEQ`
	// arm64:-`.*faceeq`,`CSET\tEQ`
	return y == x
}
