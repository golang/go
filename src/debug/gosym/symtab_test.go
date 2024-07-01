// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gosym

import (
	"fmt"
	"testing"
)

func assertString(t *testing.T, dsc, out, tgt string) {
	t.Helper()
	if out != tgt {
		t.Fatalf("Expected: %q Actual: %q for %s", tgt, out, dsc)
	}
}

func TestStandardLibPackage(t *testing.T) {
	s1 := Sym{Name: "io.(*LimitedReader).Read"}
	s2 := Sym{Name: "io.NewSectionReader"}
	assertString(t, fmt.Sprintf("package of %q", s1.Name), s1.PackageName(), "io")
	assertString(t, fmt.Sprintf("package of %q", s2.Name), s2.PackageName(), "io")
	assertString(t, fmt.Sprintf("receiver of %q", s1.Name), s1.ReceiverName(), "(*LimitedReader)")
	assertString(t, fmt.Sprintf("receiver of %q", s2.Name), s2.ReceiverName(), "")
}

func TestStandardLibPathPackage(t *testing.T) {
	s1 := Sym{Name: "debug/gosym.(*LineTable).PCToLine"}
	s2 := Sym{Name: "debug/gosym.NewTable"}
	assertString(t, fmt.Sprintf("package of %q", s1.Name), s1.PackageName(), "debug/gosym")
	assertString(t, fmt.Sprintf("package of %q", s2.Name), s2.PackageName(), "debug/gosym")
	assertString(t, fmt.Sprintf("receiver of %q", s1.Name), s1.ReceiverName(), "(*LineTable)")
	assertString(t, fmt.Sprintf("receiver of %q", s2.Name), s2.ReceiverName(), "")
}

func TestGenericNames(t *testing.T) {
	s1 := Sym{Name: "main.set[int]"}
	s2 := Sym{Name: "main.(*value[int]).get"}
	s3 := Sym{Name: "a/b.absDifference[c/d.orderedAbs[float64]]"}
	s4 := Sym{Name: "main.testfunction[.shape.int]"}
	assertString(t, fmt.Sprintf("package of %q", s1.Name), s1.PackageName(), "main")
	assertString(t, fmt.Sprintf("package of %q", s2.Name), s2.PackageName(), "main")
	assertString(t, fmt.Sprintf("package of %q", s3.Name), s3.PackageName(), "a/b")
	assertString(t, fmt.Sprintf("package of %q", s4.Name), s4.PackageName(), "main")
	assertString(t, fmt.Sprintf("receiver of %q", s1.Name), s1.ReceiverName(), "")
	assertString(t, fmt.Sprintf("receiver of %q", s2.Name), s2.ReceiverName(), "(*value[int])")
	assertString(t, fmt.Sprintf("receiver of %q", s3.Name), s3.ReceiverName(), "")
	assertString(t, fmt.Sprintf("receiver of %q", s4.Name), s4.ReceiverName(), "")
	assertString(t, fmt.Sprintf("base of %q", s1.Name), s1.BaseName(), "set[int]")
	assertString(t, fmt.Sprintf("base of %q", s2.Name), s2.BaseName(), "get")
	assertString(t, fmt.Sprintf("base of %q", s3.Name), s3.BaseName(), "absDifference[c/d.orderedAbs[float64]]")
	assertString(t, fmt.Sprintf("base of %q", s4.Name), s4.BaseName(), "testfunction[.shape.int]")
}

func TestRemotePackage(t *testing.T) {
	s1 := Sym{Name: "github.com/docker/doc.ker/pkg/mflag.(*FlagSet).PrintDefaults"}
	s2 := Sym{Name: "github.com/docker/doc.ker/pkg/mflag.PrintDefaults"}
	assertString(t, fmt.Sprintf("package of %q", s1.Name), s1.PackageName(), "github.com/docker/doc.ker/pkg/mflag")
	assertString(t, fmt.Sprintf("package of %q", s2.Name), s2.PackageName(), "github.com/docker/doc.ker/pkg/mflag")
	assertString(t, fmt.Sprintf("receiver of %q", s1.Name), s1.ReceiverName(), "(*FlagSet)")
	assertString(t, fmt.Sprintf("receiver of %q", s2.Name), s2.ReceiverName(), "")
}

func TestIssue29551(t *testing.T) {
	tests := []struct {
		sym     Sym
		pkgName string
	}{
		{Sym{goVersion: ver120, Name: "type:.eq.[9]debug/elf.intName"}, ""},
		{Sym{goVersion: ver120, Name: "type:.hash.debug/elf.ProgHeader"}, ""},
		{Sym{goVersion: ver120, Name: "type:.eq.runtime._panic"}, ""},
		{Sym{goVersion: ver120, Name: "type:.hash.struct { runtime.gList; runtime.n int32 }"}, ""},
		{Sym{goVersion: ver120, Name: "go:(*struct { sync.Mutex; math/big.table [64]math/big"}, ""},
		{Sym{goVersion: ver120, Name: "go.uber.org/zap/buffer.(*Buffer).AppendString"}, "go.uber.org/zap/buffer"},
		{Sym{goVersion: ver118, Name: "type..eq.[9]debug/elf.intName"}, ""},
		{Sym{goVersion: ver118, Name: "type..hash.debug/elf.ProgHeader"}, ""},
		{Sym{goVersion: ver118, Name: "type..eq.runtime._panic"}, ""},
		{Sym{goVersion: ver118, Name: "type..hash.struct { runtime.gList; runtime.n int32 }"}, ""},
		{Sym{goVersion: ver118, Name: "go.(*struct { sync.Mutex; math/big.table [64]math/big"}, ""},
		// unfortunate
		{Sym{goVersion: ver118, Name: "go.uber.org/zap/buffer.(*Buffer).AppendString"}, ""},
	}

	for _, tc := range tests {
		assertString(t, fmt.Sprintf("package of %q", tc.sym.Name), tc.sym.PackageName(), tc.pkgName)
	}
}

func TestIssue66313(t *testing.T) {
	tests := []struct {
		sym          Sym
		packageName  string
		receiverName string
		baseName     string
	}{
		{Sym{Name: "github.com/google/cel-go/parser/gen.(*CELLexer).reset+10c630b8"},
			"github.com/google/cel-go/parser/gen",
			"(*CELLexer)",
			"reset+10c630b8",
		},
		{Sym{Name: "ariga.io/atlas/sql/sqlclient.(*Tx).grabConn+404a5a3"},
			"ariga.io/atlas/sql/sqlclient",
			"(*Tx)",
			"grabConn+404a5a3"},
	}

	for _, tc := range tests {
		assertString(t, fmt.Sprintf("package of %q", tc.sym.Name), tc.sym.PackageName(), tc.packageName)
		assertString(t, fmt.Sprintf("receiver of %q", tc.sym.Name), tc.sym.ReceiverName(), tc.receiverName)
		assertString(t, fmt.Sprintf("package of %q", tc.sym.Name), tc.sym.BaseName(), tc.baseName)
	}
}
