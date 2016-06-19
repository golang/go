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

func TestRemotePackage(t *testing.T) {
	s1 := Sym{Name: "github.com/docker/doc.ker/pkg/mflag.(*FlagSet).PrintDefaults"}
	s2 := Sym{Name: "github.com/docker/doc.ker/pkg/mflag.PrintDefaults"}
	assertString(t, fmt.Sprintf("package of %q", s1.Name), s1.PackageName(), "github.com/docker/doc.ker/pkg/mflag")
	assertString(t, fmt.Sprintf("package of %q", s2.Name), s2.PackageName(), "github.com/docker/doc.ker/pkg/mflag")
	assertString(t, fmt.Sprintf("receiver of %q", s1.Name), s1.ReceiverName(), "(*FlagSet)")
	assertString(t, fmt.Sprintf("receiver of %q", s2.Name), s2.ReceiverName(), "")
}
