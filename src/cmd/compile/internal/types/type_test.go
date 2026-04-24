// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"bytes"
	"fmt"
	"strings"
	"testing"

	"cmd/internal/src"
)

// testTypeName is a minimal types.Object for NewNamed in tests.
type testTypeName struct {
	sym *Sym
}

func (o *testTypeName) Sym() *Sym     { return o.sym }
func (o *testTypeName) Pos() src.XPos { return src.NoXPos }
func (o *testTypeName) Type() *Type   { return nil }

// If the printer mistakenly switches to fmtTypeID while formatting an
// unexported interface method name, the method's signature is then
// printed in fmtTypeID as well, so package-local names use Pkg.Prefix
// (import path) instead of Pkg.Name. Regression test for that mistake.
func TestFormatInterfaceUnexportedMethodUsesGoModeForSignature(t *testing.T) {
	oldLocal := LocalPkg
	oldPtr, oldReg, oldMax := PtrSize, RegSize, MaxWidth
	t.Cleanup(func() {
		LocalPkg = oldLocal
		PtrSize, RegSize, MaxWidth = oldPtr, oldReg, oldMax
	})
	LocalPkg = NewPkg("main", "main")
	PtrSize = 8
	RegSize = 8
	MaxWidth = 1 << 50

	lib := NewPkg("example.com/lib", "lib")
	namedT := NewNamed(&testTypeName{sym: lib.Lookup("T")})
	// Underlying must not rely on Types[…]; InitTypes is not run in this package's tests.
	namedT.SetUnderlying(NewStruct(nil))

	sig := NewSignature(nil, nil, []*Field{NewField(src.NoXPos, nil, namedT)})
	iface := NewInterface([]*Field{NewField(src.NoXPos, LocalPkg.Lookup("unexported"), sig)})
	CalcSize(iface) // populate interface method set for printing (expandiface)

	var buf bytes.Buffer
	fmt.Fprintf(&buf, "%v", iface)
	got := buf.String()
	// Go syntax should qualify T with the package name, not the linker prefix / path.
	const want = "lib.T"
	if !strings.Contains(got, want) {
		t.Fatalf("formatted type missing %q:\n%s", want, got)
	}
	// Stale fmtTypeID mode used Prefix ("example.com/lib") instead of Name ("lib").
	if strings.Contains(got, "example.com/lib.T") {
		t.Fatalf("formatted type should not use import path to qualify T (found example.com/lib.T):\n%s", got)
	}
}

func TestSSACompare(t *testing.T) {
	a := []*Type{
		TypeInvalid,
		TypeMem,
		TypeFlags,
		TypeVoid,
		TypeInt128,
	}
	for _, x := range a {
		for _, y := range a {
			c := x.Compare(y)
			if x == y && c != CMPeq || x != y && c == CMPeq {
				t.Errorf("%s compare %s == %d\n", x.extra, y.extra, c)
			}
		}
	}
}
