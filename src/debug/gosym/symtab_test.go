// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gosym

import (
	"fmt"
	"testing"
)

func assertString(t *testing.T, dsc, out, tgt string) {
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
	symNames := []string{
		"type..eq.[9]debug/elf.intName",
		"type..hash.debug/elf.ProgHeader",
		"type..eq.runtime._panic",
		"type..hash.struct { runtime.gList; runtime.n int32 }",
		"go.(*struct { sync.Mutex; math/big.table [64]math/big",
	}

	for _, symName := range symNames {
		s := Sym{Name: symName}
		assertString(t, fmt.Sprintf("package of %q", s.Name), s.PackageName(), "")
	}
}
