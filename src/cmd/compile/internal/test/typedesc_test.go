// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"fmt"
	"internal/abi"
	"internal/testenv"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"cmd/internal/archive"
	"cmd/internal/goobj"
	"cmd/internal/objabi"
)

// TestNoalgDescriptorUpgrade checks that a type descriptor does not depend
// on whether the package first needed the type in a noalg context.
//
// A package that needs a type only as part of something marked noalg, such
// as the group type of a map, writes a descriptor without the type's hash
// and equality algorithms. If the package then needs the type for its own
// sake, writeType upgrades that descriptor in place instead of writing a
// new one. The upgraded descriptor has to come out the same as the one a
// package that never needed the noalg form writes, because the linker may
// keep either. See issue #82462.
func TestNoalgDescriptorUpgrade(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	t.Parallel()

	// A map[int]int holds its keys and its elements in
	// [abi.MapGroupSlots]int arrays within the group type, which is the
	// type the packages below reach by two different routes.
	const n = abi.MapGroupSlots
	sym := fmt.Sprintf("type:[%d]int", n)

	// noalgFirst reaches the array type through the group type of M, which
	// is marked noalg, before reaching it on its own.
	noalgFirst := fmt.Sprintf(`package p

type M map[int]int

var sinkM M
var sinkA [%[1]d]int

func useMap()   { sinkM = M{1: 2} }
func useArray() { sinkA = [%[1]d]int{} }
`, n)
	// noalgOnly reaches it only through the group type of M, so its
	// descriptor stays noalg.
	const noalgOnly = `package p

type M map[int]int

var sinkM M

func useMap() { sinkM = M{1: 2} }
`
	// algOnly reaches it on its own, and never in a noalg context.
	algOnly := fmt.Sprintf(`package p

var sinkA [%[1]d]int

func useArray() { sinkA = [%[1]d]int{} }
`, n)

	dir := t.TempDir()
	upgraded := compileSym(t, dir, "upgraded", noalgFirst, sym)
	noalg := compileSym(t, dir, "noalg", noalgOnly, sym)
	full := compileSym(t, dir, "full", algOnly, sym)

	// Guard against the test going vacuous: noalgOnly must really produce
	// a noalg descriptor, so that noalgFirst really exercises the upgrade.
	if !strings.Contains(noalg, "weakdef=true") {
		t.Fatalf("%s is not noalg when reached only through a map group type:\n%s", sym, noalg)
	}

	if upgraded != full {
		t.Errorf("%s differs depending on whether a noalg descriptor was written first\nupgraded:\n%s\nwritten in full:\n%s", sym, upgraded, full)
	}
}

// compileSym compiles src as package p in its own directory under dir, and
// returns a description of the symbol named sym: everything the object file
// records about it, except for the name.
func compileSym(t *testing.T, dir, name, src, sym string) string {
	t.Helper()

	pdir := filepath.Join(dir, name)
	if err := os.Mkdir(pdir, 0o777); err != nil {
		t.Fatal(err)
	}
	gofile := filepath.Join(pdir, "p.go")
	if err := os.WriteFile(gofile, []byte(src), 0o666); err != nil {
		t.Fatal(err)
	}
	ofile := filepath.Join(pdir, "p.o")
	out, err := testenv.Command(t, testenv.GoToolPath(t), "tool", "compile", "-p=p", "-o", ofile, gofile).CombinedOutput()
	if err != nil {
		t.Fatalf("compiling %s failed: %v\n%s", name, err, out)
	}

	f, err := os.Open(ofile)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	a, err := archive.Parse(f, false)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range a.Entries {
		if e.Type != archive.EntryGoObj {
			continue
		}
		b := make([]byte, e.Obj.Size)
		if _, err := f.ReadAt(b, e.Obj.Offset); err != nil {
			t.Fatal(err)
		}
		if s := describeSym(goobj.NewReaderFromBytes(b, false), sym); s != "" {
			return s
		}
	}
	t.Fatalf("%s: no %s symbol in %s", name, sym, ofile)
	return ""
}

// describeSym returns a description of the symbol named sym defined in r, or
// the empty string if r does not define one.
func describeSym(r *goobj.Reader, sym string) string {
	resolve := func(s goobj.SymRef) string {
		// Referenced symbols from other packages are named in the refName
		// table; the rest are indexed, in the order the reader lays them out.
		var i uint32
		switch s.PkgIdx {
		case goobj.PkgIdxInvalid:
			return ""
		case goobj.PkgIdxHashed64:
			i = s.SymIdx + uint32(r.NSym())
		case goobj.PkgIdxHashed:
			i = s.SymIdx + uint32(r.NSym()+r.NHashed64def())
		case goobj.PkgIdxNone:
			i = s.SymIdx + uint32(r.NSym()+r.NHashed64def()+r.NHasheddef())
		case goobj.PkgIdxBuiltin:
			name, _ := goobj.BuiltinName(int(s.SymIdx))
			return name
		case goobj.PkgIdxSelf:
			i = s.SymIdx
		default:
			for j := 0; j < r.NRefName(); j++ {
				if rn := r.RefName(j); rn.Sym() == s {
					return rn.Name(r)
				}
			}
			return "?"
		}
		return r.Sym(i).Name(r)
	}

	ndef := uint32(r.NSym() + r.NHashed64def() + r.NHasheddef() + r.NNonpkgdef())
	for i := uint32(0); i < ndef; i++ {
		s := r.Sym(i)
		if s.Name(r) != sym {
			continue
		}
		var b strings.Builder
		fmt.Fprintf(&b, "%v size=%d align=%d dupok=%v weakdef=%v typelink=%v\n",
			objabi.SymKind(s.Type()), s.Siz(), s.Align(), s.Dupok(), s.WeakDef(), s.Typelink())
		fmt.Fprintf(&b, "\tdata % x\n", r.Data(i))
		for _, rel := range r.Relocs(i) {
			fmt.Fprintf(&b, "\trel %d+%d t=%v %s+%d\n", rel.Off(), rel.Siz(), objabi.RelocType(rel.Type()), resolve(rel.Sym()), rel.Add())
		}
		return b.String()
	}
	return ""
}
