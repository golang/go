// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"bytes"
	"debug/dwarf"
	"debug/elf"
	"debug/pe"
	"fmt"
	"internal/platform"
	"internal/testenv"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"testing"

	intdwarf "cmd/internal/dwarf"
	objfilepkg "cmd/internal/objfile" // renamed to avoid conflict with objfile function
	"cmd/link/internal/dwtest"
)

func mustHaveDWARF(t testing.TB) {
	if !platform.ExecutableHasDWARF(runtime.GOOS, runtime.GOARCH) {
		t.Helper()
		t.Skipf("skipping on %s/%s: no DWARF symbol table in executables", runtime.GOOS, runtime.GOARCH)
	}
}

const (
	DefaultOpt = "-gcflags="
	NoOpt      = "-gcflags=-l -N"
	OptInl4    = "-gcflags=-l=4"
	OptAllInl4 = "-gcflags=all=-l=4"
)

func TestRuntimeTypesPresent(t *testing.T) {
	t.Parallel()
	testenv.MustHaveGoBuild(t)

	mustHaveDWARF(t)

	dir := t.TempDir()

	f := gobuild(t, dir, `package main; func main() { }`, NoOpt)
	defer f.Close()

	dwarf, err := f.DWARF()
	if err != nil {
		t.Fatalf("error reading DWARF: %v", err)
	}

	want := map[string]bool{
		"internal/abi.Type":          true,
		"internal/abi.ArrayType":     true,
		"internal/abi.ChanType":      true,
		"internal/abi.FuncType":      true,
		"internal/abi.PtrType":       true,
		"internal/abi.SliceType":     true,
		"internal/abi.StructType":    true,
		"internal/abi.InterfaceType": true,
		"internal/abi.ITab":          true,
	}

	found := findTypes(t, dwarf, want)
	if len(found) != len(want) {
		t.Errorf("found %v, want %v", found, want)
	}

	// Must have one of OldMapType or SwissMapType.
	want = map[string]bool{
		"internal/abi.OldMapType":   true,
		"internal/abi.SwissMapType": true,
	}
	found = findTypes(t, dwarf, want)
	if len(found) != 1 {
		t.Errorf("map type want one of %v found %v", want, found)
	}
}

func findTypes(t *testing.T, dw *dwarf.Data, want map[string]bool) (found map[string]bool) {
	found = make(map[string]bool)
	rdr := dw.Reader()
	for entry, err := rdr.Next(); entry != nil; entry, err = rdr.Next() {
		if err != nil {
			t.Fatalf("error reading DWARF: %v", err)
		}
		switch entry.Tag {
		case dwarf.TagTypedef:
			if name, ok := entry.Val(dwarf.AttrName).(string); ok && want[name] {
				found[name] = true
			}
		}
	}
	return
}

type builtFile struct {
	*objfilepkg.File
	path string
}

func gobuild(t *testing.T, dir string, testfile string, gcflags string) *builtFile {
	src := filepath.Join(dir, "test.go")
	dst := filepath.Join(dir, "out.exe")

	if err := os.WriteFile(src, []byte(testfile), 0666); err != nil {
		t.Fatal(err)
	}

	cmd := testenv.Command(t, testenv.GoToolPath(t), "build", gcflags, "-o", dst, src)
	b, err := cmd.CombinedOutput()
	if len(b) != 0 {
		t.Logf("## build output:\n%s", b)
	}
	if err != nil {
		t.Fatalf("build error: %v", err)
	}

	f, err := objfilepkg.Open(dst)
	if err != nil {
		t.Fatal(err)
	}
	return &builtFile{f, dst}
}

// Similar to gobuild() above, but uses a main package instead of a test.go file.

func gobuildTestdata(t *testing.T, pkgDir string, gcflags string) *builtFile {
	dst := filepath.Join(t.TempDir(), "out.exe")

	// Run a build with an updated GOPATH
	cmd := testenv.Command(t, testenv.GoToolPath(t), "build", gcflags, "-o", dst)
	cmd.Dir = pkgDir
	if b, err := cmd.CombinedOutput(); err != nil {
		t.Logf("build: %s\n", b)
		t.Fatalf("build error: %v", err)
	}

	f, err := objfilepkg.Open(dst)
	if err != nil {
		t.Fatal(err)
	}
	return &builtFile{f, dst}
}

// Helper to build a snippet of source for examination with dwtest.Examiner.
func gobuildAndExamine(t *testing.T, source string, gcflags string) (*dwarf.Data, *dwtest.Examiner) {
	dir := t.TempDir()

	f := gobuild(t, dir, source, gcflags)
	defer f.Close()

	d, err := f.DWARF()
	if err != nil {
		t.Fatalf("error reading DWARF in program %q: %v", source, err)
	}

	rdr := d.Reader()
	ex := &dwtest.Examiner{}
	if err := ex.Populate(rdr); err != nil {
		t.Fatalf("error populating DWARF examiner for program %q: %v", source, err)
	}

	return d, ex
}

func findSubprogramDIE(t *testing.T, ex *dwtest.Examiner, sym string) *dwarf.Entry {
	dies := ex.Named(sym)
	if len(dies) == 0 {
		t.Fatalf("unable to locate DIE for %s", sym)
	}
	if len(dies) != 1 {
		t.Fatalf("more than one %s DIE: %+v", sym, dies)
	}
	die := dies[0]

	// Vet the DIE.
	if die.Tag != dwarf.TagSubprogram {
		t.Fatalf("unexpected tag %v on %s DIE", die.Tag, sym)
	}

	return die
}

func TestEmbeddedStructMarker(t *testing.T) {
	t.Parallel()
	testenv.MustHaveGoBuild(t)

	mustHaveDWARF(t)

	const prog = `
package main

import "fmt"

type Foo struct { v int }
type Bar struct {
	Foo
	name string
}
type Baz struct {
	*Foo
	name string
}

func main() {
	bar := Bar{ Foo: Foo{v: 123}, name: "onetwothree"}
	baz := Baz{ Foo: &bar.Foo, name: "123" }
	fmt.Println(bar, baz)
}`

	want := map[string]map[string]bool{
		"main.Foo": {"v": false},
		"main.Bar": {"Foo": true, "name": false},
		"main.Baz": {"Foo": true, "name": false},
	}

	dir := t.TempDir()

	f := gobuild(t, dir, prog, NoOpt)

	defer f.Close()

	d, err := f.DWARF()
	if err != nil {
		t.Fatalf("error reading DWARF: %v", err)
	}

	rdr := d.Reader()
	for entry, err := rdr.Next(); entry != nil; entry, err = rdr.Next() {
		if err != nil {
			t.Fatalf("error reading DWARF: %v", err)
		}
		switch entry.Tag {
		case dwarf.TagStructType:
			name, ok := entry.Val(dwarf.AttrName).(string)
			if !ok {
				continue
			}
			wantMembers := want[name]
			if wantMembers == nil {
				continue
			}
			gotMembers, err := findMembers(rdr)
			if err != nil {
				t.Fatalf("error reading DWARF: %v", err)
			}

			if !reflect.DeepEqual(gotMembers, wantMembers) {
				t.Errorf("type %v: got map[member]embedded = %+v, want %+v", name, wantMembers, gotMembers)
			}
			delete(want, name)
		}
	}
	if len(want) != 0 {
		t.Errorf("failed to check all expected types: missing types = %+v", want)
	}
}

func findMembers(rdr *dwarf.Reader) (map[string]bool, error) {
	memberEmbedded := map[string]bool{}
	// TODO(hyangah): define in debug/dwarf package
	const goEmbeddedStruct = dwarf.Attr(intdwarf.DW_AT_go_embedded_field)
	for entry, err := rdr.Next(); entry != nil; entry, err = rdr.Next() {
		if err != nil {
			return nil, err
		}
		switch entry.Tag {
		case dwarf.TagMember:
			name := entry.Val(dwarf.AttrName).(string)
			embedded := entry.Val(goEmbeddedStruct).(bool)
			memberEmbedded[name] = embedded
		case 0:
			return memberEmbedded, nil
		}
	}
	return memberEmbedded, nil
}

func TestSizes(t *testing.T) {
	mustHaveDWARF(t)

	// External linking may bring in C symbols with unknown size. Skip.
	testenv.MustInternalLink(t, false)

	t.Parallel()

	// DWARF sizes should never be -1.
	// See issue #21097
	const prog = `
package main
var x func()
var y [4]func()
func main() {
	x = nil
	y[0] = nil
}
`
	dir := t.TempDir()

	f := gobuild(t, dir, prog, NoOpt)
	defer f.Close()
	d, err := f.DWARF()
	if err != nil {
		t.Fatalf("error reading DWARF: %v", err)
	}
	rdr := d.Reader()
	for entry, err := rdr.Next(); entry != nil; entry, err = rdr.Next() {
		if err != nil {
			t.Fatalf("error reading DWARF: %v", err)
		}
		switch entry.Tag {
		case dwarf.TagArrayType, dwarf.TagPointerType, dwarf.TagStructType, dwarf.TagBaseType, dwarf.TagSubroutineType, dwarf.TagTypedef:
		default:
			continue
		}
		typ, err := d.Type(entry.Offset)
		if err != nil {
			t.Fatalf("can't read type: %v", err)
		}
		if typ.Size() < 0 {
			t.Errorf("subzero size %s %s %T", typ, entry.Tag, typ)
		}
	}
}

func TestFieldOverlap(t *testing.T) {
	mustHaveDWARF(t)
	t.Parallel()

	// This test grew out of issue 21094, where specific sudog<T> DWARF types
	// had elem fields set to values instead of pointers.
	const prog = `
package main

var c chan string

func main() {
	c <- "foo"
}
`
	dir := t.TempDir()

	f := gobuild(t, dir, prog, NoOpt)
	defer f.Close()

	d, err := f.DWARF()
	if err != nil {
		t.Fatalf("error reading DWARF: %v", err)
	}

	rdr := d.Reader()
	for entry, err := rdr.Next(); entry != nil; entry, err = rdr.Next() {
		if err != nil {
			t.Fatalf("error reading DWARF: %v", err)
		}
		if entry.Tag != dwarf.TagStructType {
			continue
		}
		typ, err := d.Type(entry.Offset)
		if err != nil {
			t.Fatalf("can't read type: %v", err)
		}
		s := typ.(*dwarf.StructType)
		for i := 0; i < len(s.Field); i++ {
			end := s.Field[i].ByteOffset + s.Field[i].Type.Size()
			var limit int64
			if i == len(s.Field)-1 {
				limit = s.Size()
			} else {
				limit = s.Field[i+1].ByteOffset
			}
			if end > limit {
				name := entry.Val(dwarf.AttrName).(string)
				t.Fatalf("field %s.%s overlaps next field", name, s.Field[i].Name)
			}
		}
	}
}

func TestSubprogramDeclFileLine(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	t.Parallel()

	mustHaveDWARF(t)

	const prog = `package main
%s
func main() {}
`
	tests := []struct {
		name string
		prog string
		file string
		line int64
	}{
		{
			name: "normal",
			prog: fmt.Sprintf(prog, ""),
			file: "test.go",
			line: 3,
		},
		{
			name: "line-directive",
			prog: fmt.Sprintf(prog, "//line /foobar.go:200"),
			file: "foobar.go",
			line: 200,
		},
	}
	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			d, ex := gobuildAndExamine(t, tc.prog, NoOpt)

			maindie := findSubprogramDIE(t, ex, "main.main")

			mainIdx := ex.IdxFromOffset(maindie.Offset)

			fileIdx, fileIdxOK := maindie.Val(dwarf.AttrDeclFile).(int64)
			if !fileIdxOK {
				t.Errorf("missing or invalid DW_AT_decl_file for main")
			}
			file, err := ex.FileRef(d, mainIdx, fileIdx)
			if err != nil {
				t.Fatalf("FileRef: %v", err)
			}
			base := filepath.Base(file)
			if base != tc.file {
				t.Errorf("DW_AT_decl_file for main is %v, want %v", base, tc.file)
			}

			line, lineOK := maindie.Val(dwarf.AttrDeclLine).(int64)
			if !lineOK {
				t.Errorf("missing or invalid DW_AT_decl_line for main")
			}
			if line != tc.line {
				t.Errorf("DW_AT_decl_line for main is %v, want %d", line, tc.line)
			}
		})
	}
}

func TestVarDeclLine(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	t.Parallel()

	mustHaveDWARF(t)

	const prog = `package main
%s
func main() {

	var i int
	i = i
}
`
	tests := []struct {
		name string
		prog string
		line int64
	}{
		{
			name: "normal",
			prog: fmt.Sprintf(prog, ""),
			line: 5,
		},
		{
			name: "line-directive",
			prog: fmt.Sprintf(prog, "//line /foobar.go:200"),
			line: 202,
		},
	}
	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			_, ex := gobuildAndExamine(t, tc.prog, NoOpt)

			maindie := findSubprogramDIE(t, ex, "main.main")

			mainIdx := ex.IdxFromOffset(maindie.Offset)
			childDies := ex.Children(mainIdx)
			var iEntry *dwarf.Entry
			for _, child := range childDies {
				if child.Tag == dwarf.TagVariable && child.Val(dwarf.AttrName).(string) == "i" {
					iEntry = child
					break
				}
			}
			if iEntry == nil {
				t.Fatalf("didn't find DW_TAG_variable for i in main.main")
			}

			// Verify line/file attributes.
			line, lineOK := iEntry.Val(dwarf.AttrDeclLine).(int64)
			if !lineOK {
				t.Errorf("missing or invalid DW_AT_decl_line for i")
			}
			if line != tc.line {
				t.Errorf("DW_AT_decl_line for i is %v, want %d", line, tc.line)
			}
		})
	}
}

// TestInlinedRoutineCallFileLine tests the call file and line records for an
// inlined subroutine.
func TestInlinedRoutineCallFileLine(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	mustHaveDWARF(t)

	t.Parallel()

	const prog = `
package main

var G int

//go:noinline
func notinlined() int {
	return 42
}

func inlined() int {
	return notinlined()
}

%s
func main() {
	x := inlined()
	G = x
}
`
	tests := []struct {
		name string
		prog string
		file string // basename
		line int64
	}{
		{
			name: "normal",
			prog: fmt.Sprintf(prog, ""),
			file: "test.go",
			line: 17,
		},
		{
			name: "line-directive",
			prog: fmt.Sprintf(prog, "//line /foobar.go:200"),
			file: "foobar.go",
			line: 201,
		},
	}
	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			// Note: this is a build with "-l=4", as opposed to "-l -N". The
			// test is intended to verify DWARF that is only generated when
			// the inliner is active. We're only going to look at the DWARF for
			// main.main, however, hence we build with "-gcflags=-l=4" as opposed
			// to "-gcflags=all=-l=4".
			d, ex := gobuildAndExamine(t, tc.prog, OptInl4)

			maindie := findSubprogramDIE(t, ex, "main.main")

			// Walk main's children and pick out the inlined subroutines
			mainIdx := ex.IdxFromOffset(maindie.Offset)
			childDies := ex.Children(mainIdx)
			found := false
			for _, child := range childDies {
				if child.Tag != dwarf.TagInlinedSubroutine {
					continue
				}

				// Found an inlined subroutine.
				if found {
					t.Fatalf("Found multiple inlined subroutines, expect only one")
				}
				found = true

				// Locate abstract origin.
				ooff, originOK := child.Val(dwarf.AttrAbstractOrigin).(dwarf.Offset)
				if !originOK {
					t.Fatalf("no abstract origin attr for inlined subroutine at offset %v", child.Offset)
				}
				originDIE := ex.EntryFromOffset(ooff)
				if originDIE == nil {
					t.Fatalf("can't locate origin DIE at off %v", ooff)
				}

				// Name should check out.
				name, ok := originDIE.Val(dwarf.AttrName).(string)
				if !ok {
					t.Fatalf("no name attr for inlined subroutine at offset %v", child.Offset)
				}
				if name != "main.inlined" {
					t.Fatalf("expected inlined routine %s got %s", "main.cand", name)
				}

				// Verify that the call_file attribute for the inlined
				// instance is ok. In this case it should match the file
				// for the main routine. To do this we need to locate the
				// compilation unit DIE that encloses what we're looking
				// at; this can be done with the examiner.
				cf, cfOK := child.Val(dwarf.AttrCallFile).(int64)
				if !cfOK {
					t.Fatalf("no call_file attr for inlined subroutine at offset %v", child.Offset)
				}
				file, err := ex.FileRef(d, mainIdx, cf)
				if err != nil {
					t.Errorf("FileRef: %v", err)
					continue
				}
				base := filepath.Base(file)
				if base != tc.file {
					t.Errorf("bad call_file attribute, found '%s', want '%s'",
						file, tc.file)
				}

				// Verify that the call_line attribute for the inlined
				// instance is ok.
				cl, clOK := child.Val(dwarf.AttrCallLine).(int64)
				if !clOK {
					t.Fatalf("no call_line attr for inlined subroutine at offset %v", child.Offset)
				}
				if cl != tc.line {
					t.Errorf("bad call_line attribute, found %d, want %d", cl, tc.line)
				}
			}
			if !found {
				t.Fatalf("not enough inlined subroutines found in main.main")
			}
		})
	}
}

// TestInlinedRoutineArgsVars tests the argument and variable records for an inlined subroutine.
func TestInlinedRoutineArgsVars(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	mustHaveDWARF(t)

	t.Parallel()

	const prog = `
package main

var G int

func noinline(x int) int {
	defer func() { G += x }()
	return x
}

func cand(x, y int) int {
	return noinline(x+y) ^ (y - x)
}

func main() {
	x := cand(G*G,G|7%G)
	G = x
}
`
	// Note: this is a build with "-l=4", as opposed to "-l -N". The
	// test is intended to verify DWARF that is only generated when
	// the inliner is active. We're only going to look at the DWARF for
	// main.main, however, hence we build with "-gcflags=-l=4" as opposed
	// to "-gcflags=all=-l=4".
	_, ex := gobuildAndExamine(t, prog, OptInl4)

	maindie := findSubprogramDIE(t, ex, "main.main")

	// Walk main's children and pick out the inlined subroutines
	mainIdx := ex.IdxFromOffset(maindie.Offset)
	childDies := ex.Children(mainIdx)
	found := false
	for _, child := range childDies {
		if child.Tag != dwarf.TagInlinedSubroutine {
			continue
		}

		// Found an inlined subroutine.
		if found {
			t.Fatalf("Found multiple inlined subroutines, expect only one")
		}
		found = true

		// Locate abstract origin.
		ooff, originOK := child.Val(dwarf.AttrAbstractOrigin).(dwarf.Offset)
		if !originOK {
			t.Fatalf("no abstract origin attr for inlined subroutine at offset %v", child.Offset)
		}
		originDIE := ex.EntryFromOffset(ooff)
		if originDIE == nil {
			t.Fatalf("can't locate origin DIE at off %v", ooff)
		}

		// Name should check out.
		name, ok := originDIE.Val(dwarf.AttrName).(string)
		if !ok {
			t.Fatalf("no name attr for inlined subroutine at offset %v", child.Offset)
		}
		if name != "main.cand" {
			t.Fatalf("expected inlined routine %s got %s", "main.cand", name)
		}

		// Walk the children of the abstract subroutine. We expect
		// to see child variables there, even if (perhaps due to
		// optimization) there are no references to them from the
		// inlined subroutine DIE.
		absFcnIdx := ex.IdxFromOffset(ooff)
		absFcnChildDies := ex.Children(absFcnIdx)
		if len(absFcnChildDies) != 2 {
			t.Fatalf("expected abstract function: expected 2 children, got %d children", len(absFcnChildDies))
		}
		formalCount := 0
		for _, absChild := range absFcnChildDies {
			if absChild.Tag == dwarf.TagFormalParameter {
				formalCount += 1
				continue
			}
			t.Fatalf("abstract function child DIE: expected formal, got %v", absChild.Tag)
		}
		if formalCount != 2 {
			t.Fatalf("abstract function DIE: expected 2 formals, got %d", formalCount)
		}

		omap := make(map[dwarf.Offset]bool)

		// Walk the child variables of the inlined routine. Each
		// of them should have a distinct abstract origin-- if two
		// vars point to the same origin things are definitely broken.
		inlIdx := ex.IdxFromOffset(child.Offset)
		inlChildDies := ex.Children(inlIdx)
		for _, k := range inlChildDies {
			ooff, originOK := k.Val(dwarf.AttrAbstractOrigin).(dwarf.Offset)
			if !originOK {
				t.Fatalf("no abstract origin attr for child of inlined subroutine at offset %v", k.Offset)
			}
			if _, found := omap[ooff]; found {
				t.Fatalf("duplicate abstract origin at child of inlined subroutine at offset %v", k.Offset)
			}
			omap[ooff] = true
		}
	}
	if !found {
		t.Fatalf("not enough inlined subroutines found in main.main")
	}
}

func abstractOriginSanity(t *testing.T, pkgDir string, flags string) {
	t.Parallel()

	// Build with inlining, to exercise DWARF inlining support.
	f := gobuildTestdata(t, filepath.Join(pkgDir, "main"), flags)
	defer f.Close()

	d, err := f.DWARF()
	if err != nil {
		t.Fatalf("error reading DWARF: %v", err)
	}
	rdr := d.Reader()
	ex := dwtest.Examiner{}
	if err := ex.Populate(rdr); err != nil {
		t.Fatalf("error reading DWARF: %v", err)
	}

	// Make a pass through all DIEs looking for abstract origin
	// references.
	abscount := 0
	for i, die := range ex.DIEs() {
		// Does it have an abstract origin?
		ooff, originOK := die.Val(dwarf.AttrAbstractOrigin).(dwarf.Offset)
		if !originOK {
			continue
		}

		// All abstract origin references should be resolvable.
		abscount += 1
		originDIE := ex.EntryFromOffset(ooff)
		if originDIE == nil {
			ex.DumpEntry(i, false, 0)
			t.Fatalf("unresolved abstract origin ref in DIE at offset 0x%x\n", die.Offset)
		}

		// Suppose that DIE X has parameter/variable children {K1,
		// K2, ... KN}. If X has an abstract origin of A, then for
		// each KJ, the abstract origin of KJ should be a child of A.
		// Note that this same rule doesn't hold for non-variable DIEs.
		pidx := ex.IdxFromOffset(die.Offset)
		if pidx < 0 {
			t.Fatalf("can't locate DIE id")
		}
		kids := ex.Children(pidx)
		for _, kid := range kids {
			if kid.Tag != dwarf.TagVariable &&
				kid.Tag != dwarf.TagFormalParameter {
				continue
			}
			kooff, originOK := kid.Val(dwarf.AttrAbstractOrigin).(dwarf.Offset)
			if !originOK {
				continue
			}
			childOriginDIE := ex.EntryFromOffset(kooff)
			if childOriginDIE == nil {
				ex.DumpEntry(i, false, 0)
				t.Fatalf("unresolved abstract origin ref in DIE at offset %x", kid.Offset)
			}
			coidx := ex.IdxFromOffset(childOriginDIE.Offset)
			childOriginParent := ex.Parent(coidx)
			if childOriginParent != originDIE {
				ex.DumpEntry(i, false, 0)
				t.Fatalf("unexpected parent of abstract origin DIE at offset %v", childOriginDIE.Offset)
			}
		}
	}
	if abscount == 0 {
		t.Fatalf("no abstract origin refs found, something is wrong")
	}
}

func TestAbstractOriginSanity(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}

	mustHaveDWARF(t)
	abstractOriginSanity(t, "testdata/httptest", OptAllInl4)
}

func TestAbstractOriginSanityIssue25459(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	mustHaveDWARF(t)
	if runtime.GOARCH != "amd64" && runtime.GOARCH != "386" {
		t.Skip("skipping on not-amd64 not-386; location lists not supported")
	}

	abstractOriginSanity(t, "testdata/issue25459", DefaultOpt)
}

func TestAbstractOriginSanityIssue26237(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	mustHaveDWARF(t)
	abstractOriginSanity(t, "testdata/issue26237", DefaultOpt)
}

func TestRuntimeTypeAttrInternal(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	testenv.MustInternalLink(t, false)

	mustHaveDWARF(t)

	testRuntimeTypeAttr(t, "-ldflags=-linkmode=internal")
}

// External linking requires a host linker (https://golang.org/src/cmd/cgo/doc.go l.732)
func TestRuntimeTypeAttrExternal(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	testenv.MustHaveCGO(t)

	mustHaveDWARF(t)

	// Explicitly test external linking, for dsymutil compatibility on Darwin.
	if runtime.GOARCH == "ppc64" {
		t.Skip("-linkmode=external not supported on ppc64")
	}

	testRuntimeTypeAttr(t, "-ldflags=-linkmode=external")
}

func testRuntimeTypeAttr(t *testing.T, flags string) {
	t.Parallel()

	const prog = `
package main

import "unsafe"

type X struct{ _ int }

func main() {
	var x interface{} = &X{}
	p := *(*uintptr)(unsafe.Pointer(&x))
	print(p)
}
`
	dir := t.TempDir()

	f := gobuild(t, dir, prog, flags)
	defer f.Close()

	out, err := testenv.Command(t, f.path).CombinedOutput()
	if err != nil {
		t.Fatalf("could not run test program: %v", err)
	}
	addr, err := strconv.ParseUint(string(out), 10, 64)
	if err != nil {
		t.Fatalf("could not parse type address from program output %q: %v", out, err)
	}

	symbols, err := f.Symbols()
	if err != nil {
		t.Fatalf("error reading symbols: %v", err)
	}
	var types *objfilepkg.Sym
	for _, sym := range symbols {
		if sym.Name == "runtime.types" {
			types = &sym
			break
		}
	}
	if types == nil {
		t.Fatal("couldn't find runtime.types in symbols")
	}

	d, err := f.DWARF()
	if err != nil {
		t.Fatalf("error reading DWARF: %v", err)
	}

	rdr := d.Reader()
	ex := dwtest.Examiner{}
	if err := ex.Populate(rdr); err != nil {
		t.Fatalf("error reading DWARF: %v", err)
	}
	dies := ex.Named("*main.X")
	if len(dies) != 1 {
		t.Fatalf("wanted 1 DIE named *main.X, found %v", len(dies))
	}
	rtAttr := dies[0].Val(intdwarf.DW_AT_go_runtime_type)
	if rtAttr == nil {
		t.Fatalf("*main.X DIE had no runtime type attr. DIE: %v", dies[0])
	}

	if platform.DefaultPIE(runtime.GOOS, runtime.GOARCH, false) {
		return // everything is PIE, addresses are relocated
	}
	if rtAttr.(uint64)+types.Addr != addr {
		t.Errorf("DWARF type offset was %#x+%#x, but test program said %#x", rtAttr.(uint64), types.Addr, addr)
	}
}

func TestIssue27614(t *testing.T) {
	// Type references in debug_info should always use the DW_TAG_typedef_type
	// for the type, when that's generated.

	testenv.MustHaveGoBuild(t)

	mustHaveDWARF(t)

	t.Parallel()

	dir := t.TempDir()

	const prog = `package main

import "fmt"

type astruct struct {
	X int
}

type bstruct struct {
	X float32
}

var globalptr *astruct
var globalvar astruct
var bvar0, bvar1, bvar2 bstruct

func main() {
	fmt.Println(globalptr, globalvar, bvar0, bvar1, bvar2)
}
`

	f := gobuild(t, dir, prog, NoOpt)

	defer f.Close()

	data, err := f.DWARF()
	if err != nil {
		t.Fatal(err)
	}

	rdr := data.Reader()

	var astructTypeDIE, bstructTypeDIE, ptrastructTypeDIE *dwarf.Entry
	var globalptrDIE, globalvarDIE *dwarf.Entry
	var bvarDIE [3]*dwarf.Entry

	for {
		e, err := rdr.Next()
		if err != nil {
			t.Fatal(err)
		}
		if e == nil {
			break
		}

		name, _ := e.Val(dwarf.AttrName).(string)

		switch e.Tag {
		case dwarf.TagTypedef:
			switch name {
			case "main.astruct":
				astructTypeDIE = e
			case "main.bstruct":
				bstructTypeDIE = e
			}
		case dwarf.TagPointerType:
			if name == "*main.astruct" {
				ptrastructTypeDIE = e
			}
		case dwarf.TagVariable:
			switch name {
			case "main.globalptr":
				globalptrDIE = e
			case "main.globalvar":
				globalvarDIE = e
			default:
				const bvarprefix = "main.bvar"
				if strings.HasPrefix(name, bvarprefix) {
					i, _ := strconv.Atoi(name[len(bvarprefix):])
					bvarDIE[i] = e
				}
			}
		}
	}

	typedieof := func(e *dwarf.Entry) dwarf.Offset {
		return e.Val(dwarf.AttrType).(dwarf.Offset)
	}

	if off := typedieof(ptrastructTypeDIE); off != astructTypeDIE.Offset {
		t.Errorf("type attribute of *main.astruct references %#x, not main.astruct DIE at %#x\n", off, astructTypeDIE.Offset)
	}

	if off := typedieof(globalptrDIE); off != ptrastructTypeDIE.Offset {
		t.Errorf("type attribute of main.globalptr references %#x, not *main.astruct DIE at %#x\n", off, ptrastructTypeDIE.Offset)
	}

	if off := typedieof(globalvarDIE); off != astructTypeDIE.Offset {
		t.Errorf("type attribute of main.globalvar1 references %#x, not main.astruct DIE at %#x\n", off, astructTypeDIE.Offset)
	}

	for i := range bvarDIE {
		if off := typedieof(bvarDIE[i]); off != bstructTypeDIE.Offset {
			t.Errorf("type attribute of main.bvar%d references %#x, not main.bstruct DIE at %#x\n", i, off, bstructTypeDIE.Offset)
		}
	}
}

func TestStaticTmp(t *testing.T) {
	// Checks that statictmp variables do not appear in debug_info or the
	// symbol table.
	// Also checks that statictmp variables do not collide with user defined
	// variables (issue #25113)

	testenv.MustHaveGoBuild(t)

	mustHaveDWARF(t)

	t.Parallel()

	dir := t.TempDir()

	const prog = `package main

var stmp_0 string
var a []int

func init() {
	a = []int{ 7 }
}

func main() {
	println(a[0])
}
`

	f := gobuild(t, dir, prog, NoOpt)

	defer f.Close()

	d, err := f.DWARF()
	if err != nil {
		t.Fatalf("error reading DWARF: %v", err)
	}

	rdr := d.Reader()
	for {
		e, err := rdr.Next()
		if err != nil {
			t.Fatal(err)
		}
		if e == nil {
			break
		}
		if e.Tag != dwarf.TagVariable {
			continue
		}
		name, ok := e.Val(dwarf.AttrName).(string)
		if !ok {
			continue
		}
		if strings.Contains(name, "stmp") {
			t.Errorf("statictmp variable found in debug_info: %s at %x", name, e.Offset)
		}
	}

	// When external linking, we put all symbols in the symbol table (so the
	// external linker can find them). Skip the symbol table check.
	// TODO: maybe there is some way to tell the external linker not to put
	// those symbols in the executable's symbol table? Prefix the symbol name
	// with "." or "L" to pretend it is a label?
	if !testenv.CanInternalLink(false) {
		return
	}

	syms, err := f.Symbols()
	if err != nil {
		t.Fatalf("error reading symbols: %v", err)
	}
	for _, sym := range syms {
		if strings.Contains(sym.Name, "stmp") {
			t.Errorf("statictmp variable found in symbol table: %s", sym.Name)
		}
	}
}

func TestPackageNameAttr(t *testing.T) {
	const dwarfAttrGoPackageName = dwarf.Attr(0x2905)
	const dwarfGoLanguage = 22

	testenv.MustHaveGoBuild(t)

	mustHaveDWARF(t)

	t.Parallel()

	dir := t.TempDir()

	const prog = "package main\nfunc main() {\nprintln(\"hello world\")\n}\n"

	f := gobuild(t, dir, prog, NoOpt)

	defer f.Close()

	d, err := f.DWARF()
	if err != nil {
		t.Fatalf("error reading DWARF: %v", err)
	}

	rdr := d.Reader()
	runtimeUnitSeen := false
	for {
		e, err := rdr.Next()
		if err != nil {
			t.Fatal(err)
		}
		if e == nil {
			break
		}
		if e.Tag != dwarf.TagCompileUnit {
			continue
		}
		if lang, _ := e.Val(dwarf.AttrLanguage).(int64); lang != dwarfGoLanguage {
			continue
		}

		pn, ok := e.Val(dwarfAttrGoPackageName).(string)
		if !ok {
			name, _ := e.Val(dwarf.AttrName).(string)
			t.Errorf("found compile unit without package name: %s", name)

		}
		if pn == "" {
			name, _ := e.Val(dwarf.AttrName).(string)
			t.Errorf("found compile unit with empty package name: %s", name)
		} else {
			if pn == "runtime" {
				runtimeUnitSeen = true
			}
		}
	}

	// Something is wrong if there's no runtime compilation unit.
	if !runtimeUnitSeen {
		t.Errorf("no package name for runtime unit")
	}
}

func TestMachoIssue32233(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	testenv.MustHaveCGO(t)

	if runtime.GOOS != "darwin" {
		t.Skip("skipping; test only interesting on darwin")
	}

	f := gobuildTestdata(t, "testdata/issue32233/main", DefaultOpt)
	f.Close()
}

func TestWindowsIssue36495(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	if runtime.GOOS != "windows" {
		t.Skip("skipping: test only on windows")
	}

	dir := t.TempDir()

	prog := `
package main

import "fmt"

func main() {
  fmt.Println("Hello World")
}`
	f := gobuild(t, dir, prog, NoOpt)
	defer f.Close()
	exe, err := pe.Open(f.path)
	if err != nil {
		t.Fatalf("error opening pe file: %v", err)
	}
	defer exe.Close()
	dw, err := exe.DWARF()
	if err != nil {
		t.Fatalf("error parsing DWARF: %v", err)
	}
	rdr := dw.Reader()
	for {
		e, err := rdr.Next()
		if err != nil {
			t.Fatalf("error reading DWARF: %v", err)
		}
		if e == nil {
			break
		}
		if e.Tag != dwarf.TagCompileUnit {
			continue
		}
		lnrdr, err := dw.LineReader(e)
		if err != nil {
			t.Fatalf("error creating DWARF line reader: %v", err)
		}
		if lnrdr != nil {
			var lne dwarf.LineEntry
			for {
				err := lnrdr.Next(&lne)
				if err == io.EOF {
					break
				}
				if err != nil {
					t.Fatalf("error reading next DWARF line: %v", err)
				}
				if strings.Contains(lne.File.Name, `\`) {
					t.Errorf("filename should not contain backslash: %v", lne.File.Name)
				}
			}
		}
		rdr.SkipChildren()
	}
}

func TestIssue38192(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	mustHaveDWARF(t)

	t.Parallel()

	// Build a test program that contains a translation unit whose
	// text (from am assembly source) contains only a single instruction.
	f := gobuildTestdata(t, "testdata/issue38192", DefaultOpt)
	defer f.Close()

	// Open the resulting binary and examine the DWARF it contains.
	// Look for the function of interest ("main.singleInstruction")
	// and verify that the line table has an entry not just for the
	// single instruction but also a dummy instruction following it,
	// so as to test that whoever is emitting the DWARF doesn't
	// emit an end-sequence op immediately after the last instruction
	// in the translation unit.
	//
	// NB: another way to write this test would have been to run the
	// resulting executable under GDB, set a breakpoint in
	// "main.singleInstruction", then verify that GDB displays the
	// correct line/file information.  Given the headache and flakiness
	// associated with GDB-based tests these days, a direct read of
	// the line table seems more desirable.
	rows := []dwarf.LineEntry{}
	dw, err := f.DWARF()
	if err != nil {
		t.Fatalf("error parsing DWARF: %v", err)
	}
	rdr := dw.Reader()
	for {
		e, err := rdr.Next()
		if err != nil {
			t.Fatalf("error reading DWARF: %v", err)
		}
		if e == nil {
			break
		}
		if e.Tag != dwarf.TagCompileUnit {
			continue
		}
		// NB: there can be multiple compile units named "main".
		name := e.Val(dwarf.AttrName).(string)
		if name != "main" {
			continue
		}
		lnrdr, err := dw.LineReader(e)
		if err != nil {
			t.Fatalf("error creating DWARF line reader: %v", err)
		}
		if lnrdr != nil {
			var lne dwarf.LineEntry
			for {
				err := lnrdr.Next(&lne)
				if err == io.EOF {
					break
				}
				if err != nil {
					t.Fatalf("error reading next DWARF line: %v", err)
				}
				if !strings.HasSuffix(lne.File.Name, "ld/testdata/issue38192/oneline.s") {
					continue
				}
				rows = append(rows, lne)
			}
		}
		rdr.SkipChildren()
	}
	f.Close()

	// Make sure that:
	// - main.singleInstruction appears in the line table
	// - more than one PC value appears the line table for
	//   that compilation unit.
	// - at least one row has the correct line number (8)
	pcs := make(map[uint64]bool)
	line8seen := false
	for _, r := range rows {
		pcs[r.Address] = true
		if r.Line == 8 {
			line8seen = true
		}
	}
	failed := false
	if len(pcs) < 2 {
		failed = true
		t.Errorf("not enough line table rows for main.singleInstruction (got %d, wanted > 1", len(pcs))
	}
	if !line8seen {
		failed = true
		t.Errorf("line table does not contain correct line for main.singleInstruction")
	}
	if !failed {
		return
	}
	for i, r := range rows {
		t.Logf("row %d: A=%x F=%s L=%d\n", i, r.Address, r.File.Name, r.Line)
	}
}

func TestIssue39757(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	mustHaveDWARF(t)

	t.Parallel()

	// In this bug the DWARF line table contents for the last couple of
	// instructions in a function were incorrect (bad file/line). This
	// test verifies that all of the line table rows for a function
	// of interest have the same file (no "autogenerated").
	//
	// Note: the function in this test was written with an eye towards
	// ensuring that there are no inlined routines from other packages
	// (which could introduce other source files into the DWARF); it's
	// possible that at some point things could evolve in the
	// compiler/runtime in ways that aren't happening now, so this
	// might be something to check for if it does start failing.

	f := gobuildTestdata(t, "testdata/issue39757", DefaultOpt)
	defer f.Close()

	syms, err := f.Symbols()
	if err != nil {
		t.Fatal(err)
	}

	var addr uint64
	for _, sym := range syms {
		if sym.Name == "main.main" {
			addr = sym.Addr
			break
		}
	}
	if addr == 0 {
		t.Fatal("cannot find main.main in symbols")
	}

	// Open the resulting binary and examine the DWARF it contains.
	// Look for the function of interest ("main.main")
	// and verify that all line table entries show the same source
	// file.
	dw, err := f.DWARF()
	if err != nil {
		t.Fatalf("error parsing DWARF: %v", err)
	}
	rdr := dw.Reader()
	ex := &dwtest.Examiner{}
	if err := ex.Populate(rdr); err != nil {
		t.Fatalf("error reading DWARF: %v", err)
	}

	maindie := findSubprogramDIE(t, ex, "main.main")

	// Collect the start/end PC for main.main. The format/class of the
	// high PC attr may vary depending on which DWARF version we're generating;
	// invoke a helper to handle the various possibilities.
	// the low PC as opposed to an address; allow for both possibilities.
	lowpc, highpc, perr := dwtest.SubprogLoAndHighPc(maindie)
	if perr != nil {
		t.Fatalf("main.main DIE malformed: %v", perr)
	}
	t.Logf("lo=0x%x hi=0x%x\n", lowpc, highpc)

	// Now read the line table for the 'main' compilation unit.
	mainIdx := ex.IdxFromOffset(maindie.Offset)
	cuentry := ex.Parent(mainIdx)
	if cuentry == nil {
		t.Fatalf("main.main DIE appears orphaned")
	}
	lnrdr, lerr := dw.LineReader(cuentry)
	if lerr != nil {
		t.Fatalf("error creating DWARF line reader: %v", err)
	}
	if lnrdr == nil {
		t.Fatalf("no line table for main.main compilation unit")
	}
	rows := []dwarf.LineEntry{}
	mainrows := 0
	var lne dwarf.LineEntry
	for {
		err := lnrdr.Next(&lne)
		if err == io.EOF {
			break
		}
		rows = append(rows, lne)
		if err != nil {
			t.Fatalf("error reading next DWARF line: %v", err)
		}
		if lne.Address < lowpc || lne.Address > highpc {
			continue
		}
		if !strings.HasSuffix(lne.File.Name, "issue39757main.go") {
			t.Errorf("found row with file=%s (not issue39757main.go)", lne.File.Name)
		}
		mainrows++
	}
	f.Close()

	// Make sure we saw a few rows.
	if mainrows < 3 {
		t.Errorf("not enough line table rows for main.main (got %d, wanted > 3", mainrows)
		for i, r := range rows {
			t.Logf("row %d: A=%x F=%s L=%d\n", i, r.Address, r.File.Name, r.Line)
		}
	}
}

func TestIssue42484(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	testenv.MustInternalLink(t, false) // Avoid spurious failures from external linkers.

	mustHaveDWARF(t)

	t.Parallel()

	f := gobuildTestdata(t, "testdata/issue42484", NoOpt)

	var lastAddr uint64
	var lastFile string
	var lastLine int

	dw, err := f.DWARF()
	if err != nil {
		t.Fatalf("error parsing DWARF: %v", err)
	}
	rdr := dw.Reader()
	for {
		e, err := rdr.Next()
		if err != nil {
			t.Fatalf("error reading DWARF: %v", err)
		}
		if e == nil {
			break
		}
		if e.Tag != dwarf.TagCompileUnit {
			continue
		}
		lnrdr, err := dw.LineReader(e)
		if err != nil {
			t.Fatalf("error creating DWARF line reader: %v", err)
		}
		if lnrdr != nil {
			var lne dwarf.LineEntry
			for {
				err := lnrdr.Next(&lne)
				if err == io.EOF {
					break
				}
				if err != nil {
					t.Fatalf("error reading next DWARF line: %v", err)
				}
				if lne.EndSequence {
					continue
				}
				if lne.Address == lastAddr && (lne.File.Name != lastFile || lne.Line != lastLine) {
					t.Errorf("address %#x is assigned to both %s:%d and %s:%d", lastAddr, lastFile, lastLine, lne.File.Name, lne.Line)
				}
				lastAddr = lne.Address
				lastFile = lne.File.Name
				lastLine = lne.Line
			}
		}
		rdr.SkipChildren()
	}
	f.Close()
}

// processParams examines the formal parameter children of subprogram
// DIE "die" using the explorer "ex" and returns a string that
// captures the name, order, and classification of the subprogram's
// input and output parameters. For example, for the go function
//
//	func foo(i1 int, f1 float64) (string, bool) {
//
// this function would return a string something like
//
//	i1:0:1 f1:1:1 ~r0:2:2 ~r1:3:2
//
// where each chunk above is of the form NAME:ORDER:INOUTCLASSIFICATION
func processParams(die *dwarf.Entry, ex *dwtest.Examiner) string {
	// Values in the returned map are of the form <order>:<varparam>
	// where order is the order within the child DIE list of the
	// param, and <varparam> is an integer:
	//
	//  -1: varparm attr not found
	//   1: varparm found with value false
	//   2: varparm found with value true
	//
	foundParams := make(map[string]string)

	// Walk the subprogram DIE's children looking for params.
	pIdx := ex.IdxFromOffset(die.Offset)
	childDies := ex.Children(pIdx)
	idx := 0
	for _, child := range childDies {
		if child.Tag == dwarf.TagFormalParameter {
			// NB: a setting of DW_AT_variable_parameter indicates
			// that the param in question is an output parameter; we
			// want to see this attribute set to TRUE for all Go
			// return params. It would be OK to have it missing for
			// input parameters, but for the moment we verify that the
			// attr is present but set to false.
			st := -1
			if vp, ok := child.Val(dwarf.AttrVarParam).(bool); ok {
				if vp {
					st = 2
				} else {
					st = 1
				}
			}
			if name, ok := child.Val(dwarf.AttrName).(string); ok {
				foundParams[name] = fmt.Sprintf("%d:%d", idx, st)
				idx++
			}
		}
	}

	found := make([]string, 0, len(foundParams))
	for k, v := range foundParams {
		found = append(found, fmt.Sprintf("%s:%s", k, v))
	}
	sort.Strings(found)

	return fmt.Sprintf("%+v", found)
}

func TestOutputParamAbbrevAndAttr(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	mustHaveDWARF(t)
	t.Parallel()

	// This test verifies that the compiler is selecting the correct
	// DWARF abbreviation for output parameters, and that the
	// variable parameter attribute is correct for in-params and
	// out-params.

	const prog = `
package main

//go:noinline
func ABC(c1, c2, c3 int, d1, d2, d3, d4 string, f1, f2, f3 float32, g1 [1024]int) (r1 int, r2 int, r3 [1024]int, r4 byte, r5 string, r6 float32) {
	g1[0] = 6
	r1, r2, r3, r4, r5, r6 = c3, c2+c1, g1, 'a', d1+d2+d3+d4, f1+f2+f3
	return
}

func main() {
	a := [1024]int{}
	v1, v2, v3, v4, v5, v6 := ABC(1, 2, 3, "a", "b", "c", "d", 1.0, 2.0, 1.0, a)
	println(v1, v2, v3[0], v4, v5, v6)
}
`
	_, ex := gobuildAndExamine(t, prog, NoOpt)

	abcdie := findSubprogramDIE(t, ex, "main.ABC")

	// Call a helper to collect param info.
	found := processParams(abcdie, ex)

	// Make sure we see all of the expected params in the proper
	// order, that they have the varparam attr, and the varparam is
	// set for the returns.
	expected := "[c1:0:1 c2:1:1 c3:2:1 d1:3:1 d2:4:1 d3:5:1 d4:6:1 f1:7:1 f2:8:1 f3:9:1 g1:10:1 r1:11:2 r2:12:2 r3:13:2 r4:14:2 r5:15:2 r6:16:2]"
	if found != expected {
		t.Errorf("param check failed, wanted:\n%s\ngot:\n%s\n",
			expected, found)
	}
}

func TestDictIndex(t *testing.T) {
	// Check that variables with a parametric type have a dictionary index
	// attribute and that types that are only referenced through dictionaries
	// have DIEs.
	testenv.MustHaveGoBuild(t)

	mustHaveDWARF(t)
	t.Parallel()

	const prog = `
package main

import "fmt"

type CustomInt int

func testfn[T any](arg T) {
	var mapvar = make(map[int]T)
	mapvar[0] = arg
	fmt.Println(arg, mapvar)
}

func main() {
	testfn(CustomInt(3))
}
`

	dir := t.TempDir()
	f := gobuild(t, dir, prog, NoOpt)
	defer f.Close()

	d, err := f.DWARF()
	if err != nil {
		t.Fatalf("error reading DWARF: %v", err)
	}

	rdr := d.Reader()
	found := false
	for entry, err := rdr.Next(); entry != nil; entry, err = rdr.Next() {
		if err != nil {
			t.Fatalf("error reading DWARF: %v", err)
		}
		name, _ := entry.Val(dwarf.AttrName).(string)
		if strings.HasPrefix(name, "main.testfn") {
			found = true
			break
		}
	}

	if !found {
		t.Fatalf("could not find main.testfn")
	}

	offs := []dwarf.Offset{}
	for entry, err := rdr.Next(); entry != nil; entry, err = rdr.Next() {
		if err != nil {
			t.Fatalf("error reading DWARF: %v", err)
		}
		if entry.Tag == 0 {
			break
		}
		name, _ := entry.Val(dwarf.AttrName).(string)
		switch name {
		case "arg", "mapvar":
			offs = append(offs, entry.Val(dwarf.AttrType).(dwarf.Offset))
		}
	}
	if len(offs) != 2 {
		t.Errorf("wrong number of variables found in main.testfn %d", len(offs))
	}
	for _, off := range offs {
		rdr.Seek(off)
		entry, err := rdr.Next()
		if err != nil {
			t.Fatalf("error reading DWARF: %v", err)
		}
		if _, ok := entry.Val(intdwarf.DW_AT_go_dict_index).(int64); !ok {
			t.Errorf("could not find DW_AT_go_dict_index attribute offset %#x (%T)", off, entry.Val(intdwarf.DW_AT_go_dict_index))
		}
	}

	rdr.Seek(0)
	ex := dwtest.Examiner{}
	if err := ex.Populate(rdr); err != nil {
		t.Fatalf("error reading DWARF: %v", err)
	}
	for _, typeName := range []string{"main.CustomInt", "map[int]main.CustomInt"} {
		dies := ex.Named(typeName)
		if len(dies) != 1 {
			t.Errorf("wanted 1 DIE named %s, found %v", typeName, len(dies))
		}
		if dies[0].Val(intdwarf.DW_AT_go_runtime_type).(uint64) == 0 {
			t.Errorf("type %s does not have DW_AT_go_runtime_type", typeName)
		}
	}
}

func TestOptimizedOutParamHandling(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	mustHaveDWARF(t)
	t.Parallel()

	// This test is intended to verify that the compiler emits DWARF
	// DIE entries for all input and output parameters, and that:
	//
	//   - attributes are set correctly for output params,
	//   - things appear in the proper order
	//   - things work properly for both register-resident
	//     params and params passed on the stack
	//   - things work for both referenced and unreferenced params
	//   - things work for named return values un-named return vals
	//
	// The scenarios below don't cover all possible permutations and
	// combinations, but they hit a bunch of the high points.

	const prog = `
package main

// First testcase. All input params in registers, all params used.

//go:noinline
func tc1(p1, p2 int, p3 string) (int, string) {
	return p1 + p2, p3 + "foo"
}

// Second testcase. Some params in registers, some on stack.

//go:noinline
func tc2(p1 int, p2 [128]int, p3 string) (int, string, [128]int) {
	return p1 + p2[p1], p3 + "foo", [128]int{p1}
}

// Third testcase. Named return params.

//go:noinline
func tc3(p1 int, p2 [128]int, p3 string) (r1 int, r2 bool, r3 string, r4 [128]int) {
	if p1 == 101 {
		r1 = p1 + p2[p1]
		r2 = p3 == "foo"
		r4 = [128]int{p1}
		return
	} else {
		return p1 - p2[p1+3], false, "bar", [128]int{p1 + 2}
	}
}

// Fourth testcase. Some thing are used, some are unused.

//go:noinline
func tc4(p1, p1un int, p2, p2un [128]int, p3, p3un string) (r1 int, r1un int, r2 bool, r3 string, r4, r4un [128]int) {
	if p1 == 101 {
		r1 = p1 + p2[p2[0]]
		r2 = p3 == "foo"
		r4 = [128]int{p1}
		return
	} else {
		return p1, -1, true, "plex", [128]int{p1 + 2}, [128]int{-1}
	}
}

func main() {
	{
		r1, r2 := tc1(3, 4, "five")
		println(r1, r2)
	}
	{
		x := [128]int{9}
		r1, r2, r3 := tc2(3, x, "five")
		println(r1, r2, r3[0])
	}
	{
		x := [128]int{9}
		r1, r2, r3, r4 := tc3(3, x, "five")
		println(r1, r2, r3, r4[0])
	}
	{
		x := [128]int{3}
		y := [128]int{7}
		r1, r1u, r2, r3, r4, r4u := tc4(0, 1, x, y, "a", "b")
		println(r1, r1u, r2, r3, r4[0], r4u[1])
	}

}
`
	_, ex := gobuildAndExamine(t, prog, DefaultOpt)

	testcases := []struct {
		tag      string
		expected string
	}{
		{
			tag:      "tc1",
			expected: "[p1:0:1 p2:1:1 p3:2:1 ~r0:3:2 ~r1:4:2]",
		},
		{
			tag:      "tc2",
			expected: "[p1:0:1 p2:1:1 p3:2:1 ~r0:3:2 ~r1:4:2 ~r2:5:2]",
		},
		{
			tag:      "tc3",
			expected: "[p1:0:1 p2:1:1 p3:2:1 r1:3:2 r2:4:2 r3:5:2 r4:6:2]",
		},
		{
			tag:      "tc4",
			expected: "[p1:0:1 p1un:1:1 p2:2:1 p2un:3:1 p3:4:1 p3un:5:1 r1:6:2 r1un:7:2 r2:8:2 r3:9:2 r4:10:2 r4un:11:2]",
		},
	}

	for _, tc := range testcases {
		// Locate the proper DIE
		which := fmt.Sprintf("main.%s", tc.tag)
		die := findSubprogramDIE(t, ex, which)

		// Examine params for this subprogram.
		foundParams := processParams(die, ex)
		if foundParams != tc.expected {
			t.Errorf("check failed for testcase %s -- wanted:\n%s\ngot:%s\n",
				tc.tag, tc.expected, foundParams)
		}
	}
}
func TestIssue54320(t *testing.T) {
	// Check that when trampolines are used, the DWARF LPT is correctly
	// emitted in the final binary
	testenv.MustHaveGoBuild(t)

	mustHaveDWARF(t)

	t.Parallel()

	const prog = `
package main

import "fmt"

func main() {
	fmt.Printf("Hello world\n");
}
`

	dir := t.TempDir()
	f := gobuild(t, dir, prog, "-ldflags=-debugtramp=2")
	defer f.Close()

	d, err := f.DWARF()
	if err != nil {
		t.Fatalf("error reading DWARF: %v", err)
	}

	rdr := d.Reader()
	found := false
	var entry *dwarf.Entry
	for entry, err = rdr.Next(); entry != nil; entry, err = rdr.Next() {
		if err != nil {
			t.Fatalf("error reading DWARF: %v", err)
		}
		if entry.Tag != dwarf.TagCompileUnit {
			continue
		}
		name, _ := entry.Val(dwarf.AttrName).(string)
		if name == "main" {
			found = true
			break
		}
		rdr.SkipChildren()
	}

	if !found {
		t.Fatalf("could not find main compile unit")
	}
	lr, err := d.LineReader(entry)
	if err != nil {
		t.Fatalf("error obtaining linereader: %v", err)
	}

	var le dwarf.LineEntry
	found = false
	for {
		if err := lr.Next(&le); err != nil {
			if err == io.EOF {
				break
			}
			t.Fatalf("error reading linentry: %v", err)
		}
		// check LE contains an entry to test.go
		if le.File == nil {
			continue
		}
		file := filepath.Base(le.File.Name)
		if file == "test.go" {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("no LPT entries for test.go")
	}
}

const zeroSizedVarProg = `
package main

import (
	"fmt"
)

func main() {
	zeroSizedVariable := struct{}{}
	fmt.Println(zeroSizedVariable)
}
`

func TestZeroSizedVariable(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	mustHaveDWARF(t)
	t.Parallel()

	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}

	// This test verifies that the compiler emits DIEs for zero sized variables
	// (for example variables of type 'struct {}').
	// See go.dev/issues/54615.

	for _, opt := range []string{NoOpt, DefaultOpt} {
		opt := opt
		t.Run(opt, func(t *testing.T) {
			_, ex := gobuildAndExamine(t, zeroSizedVarProg, opt)

			// Locate the main.zeroSizedVariable DIE
			abcs := ex.Named("zeroSizedVariable")
			if len(abcs) == 0 {
				t.Fatalf("unable to locate DIE for zeroSizedVariable")
			}
			if len(abcs) != 1 {
				t.Fatalf("more than one zeroSizedVariable DIE")
			}
		})
	}
}

func TestConsistentGoKindAndRuntimeType(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	mustHaveDWARF(t)
	t.Parallel()

	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}

	// Ensure that if we emit a "go runtime type" attr on a type DIE,
	// we also include the "go kind" attribute. See issue #64231.
	_, ex := gobuildAndExamine(t, zeroSizedVarProg, DefaultOpt)

	// Walk all dies.
	typesChecked := 0
	failures := 0
	for _, die := range ex.DIEs() {
		// For any type DIE with DW_AT_go_runtime_type set...
		rtt, hasRT := die.Val(intdwarf.DW_AT_go_runtime_type).(uint64)
		if !hasRT || rtt == 0 {
			continue
		}
		// ... except unsafe.Pointer...
		if name, _ := die.Val(intdwarf.DW_AT_name).(string); name == "unsafe.Pointer" {
			continue
		}
		typesChecked++
		// ... we want to see a meaningful DW_AT_go_kind value.
		if val, ok := die.Val(intdwarf.DW_AT_go_kind).(int64); !ok || val == 0 {
			failures++
			// dump DIEs for first 10 failures.
			if failures <= 10 {
				idx := ex.IdxFromOffset(die.Offset)
				t.Logf("type DIE has DW_AT_go_runtime_type but invalid DW_AT_go_kind:\n")
				ex.DumpEntry(idx, false, 0)
			}
			t.Errorf("bad type DIE at offset %d\n", die.Offset)
		}
	}
	if typesChecked == 0 {
		t.Fatalf("something went wrong, 0 types checked")
	} else {
		t.Logf("%d types checked\n", typesChecked)
	}
}

func TestIssue72053(t *testing.T) {
	if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" {
		t.Skip("skipping test: requires ELF binary and amd64 arch")
	}

	testenv.MustHaveGoBuild(t)

	mustHaveDWARF(t)

	t.Parallel()

	dir := t.TempDir()

	const prog = `package main

import (
		"fmt"
		"strings"
)

func main() {
		u := Address{Addr: "127.0.0.1"}
		fmt.Println(u) // line 10
}

type Address struct {
		TLS  bool
		Addr string
}

func (a Address) String() string {
		sb := new(strings.Builder)
		sb.WriteString(a.Addr)
		return sb.String()
}
`

	bf := gobuild(t, dir, prog, NoOpt)

	defer bf.Close()

	f, err := elf.Open(bf.path)
	if err != nil {
		t.Fatal(err)
	}

	dwrf, err := f.DWARF()
	if err != nil {
		t.Fatal(err)
	}

	rdr := dwrf.Reader()

	found := false
	for {
		e, err := rdr.Next()
		if err != nil {
			t.Fatal(err)
		}
		if e == nil {
			break
		}

		name, _ := e.Val(dwarf.AttrName).(string)

		if e.Tag == dwarf.TagSubprogram && name == "main.Address.String" {
			found = true
			continue
		}

		if found && name == "a" {
			loc := e.AttrField(dwarf.AttrLocation)
			if loc != nil {
				switch loc.Class {
				case dwarf.ClassLocListPtr:
					offset := loc.Val.(int64)
					buf := make([]byte, 48)
					s := f.Section(".debug_loc")
					if s == nil {
						t.Fatal("could not find debug_loc section")
					}
					d := s.Open()
					d.Seek(offset, io.SeekStart)
					d.Read(buf)

					// DW_OP_reg0 DW_OP_piece 0x1 DW_OP_piece 0x7 DW_OP_reg3 DW_OP_piece 0x8 DW_OP_reg2 DW_OP_piece 0x8
					expected := []byte{
						0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
						0xa0, 0x2c, 0x49, 0x00, 0x00, 0x00, 0x00, 0x00,
						0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
						0x1f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
						0x0b, 0x00, 0x50, 0x93, 0x01, 0x93, 0x07, 0x53,
						0x93, 0x08, 0x52, 0x93, 0x08, 0x1f, 0x00, 0x00,
					}

					if !bytes.Equal(buf, expected) {
						t.Fatal("unexpected DWARF sequence found")
					}
				}
			} else {
				t.Fatal("unable to find expected DWARF location list")
			}
			break
		}
	}
	if !found {
		t.Fatal("unable to find expected DWARF location list")
	}
}
