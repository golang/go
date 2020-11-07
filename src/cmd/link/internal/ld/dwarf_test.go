// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	intdwarf "cmd/internal/dwarf"
	objfilepkg "cmd/internal/objfile" // renamed to avoid conflict with objfile function
	"debug/dwarf"
	"debug/pe"
	"errors"
	"fmt"
	"internal/testenv"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"testing"
)

const (
	DefaultOpt = "-gcflags="
	NoOpt      = "-gcflags=-l -N"
	OptInl4    = "-gcflags=-l=4"
	OptAllInl4 = "-gcflags=all=-l=4"
)

func TestRuntimeTypesPresent(t *testing.T) {
	t.Parallel()
	testenv.MustHaveGoBuild(t)

	if runtime.GOOS == "plan9" {
		t.Skip("skipping on plan9; no DWARF symbol table in executables")
	}

	dir, err := ioutil.TempDir("", "TestRuntimeTypesPresent")
	if err != nil {
		t.Fatalf("could not create directory: %v", err)
	}
	defer os.RemoveAll(dir)

	f := gobuild(t, dir, `package main; func main() { }`, NoOpt)
	defer f.Close()

	dwarf, err := f.DWARF()
	if err != nil {
		t.Fatalf("error reading DWARF: %v", err)
	}

	want := map[string]bool{
		"runtime._type":         true,
		"runtime.arraytype":     true,
		"runtime.chantype":      true,
		"runtime.functype":      true,
		"runtime.maptype":       true,
		"runtime.ptrtype":       true,
		"runtime.slicetype":     true,
		"runtime.structtype":    true,
		"runtime.interfacetype": true,
		"runtime.itab":          true,
		"runtime.imethod":       true,
	}

	found := findTypes(t, dwarf, want)
	if len(found) != len(want) {
		t.Errorf("found %v, want %v", found, want)
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

	if err := ioutil.WriteFile(src, []byte(testfile), 0666); err != nil {
		t.Fatal(err)
	}

	cmd := exec.Command(testenv.GoToolPath(t), "build", gcflags, "-o", dst, src)
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

// Similar to gobuild() above, but uses a main package instead of a test.go file.

func gobuildTestdata(t *testing.T, tdir string, pkgDir string, gcflags string) *builtFile {
	dst := filepath.Join(tdir, "out.exe")

	// Run a build with an updated GOPATH
	cmd := exec.Command(testenv.GoToolPath(t), "build", gcflags, "-o", dst)
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

func TestEmbeddedStructMarker(t *testing.T) {
	t.Parallel()
	testenv.MustHaveGoBuild(t)

	if runtime.GOOS == "plan9" {
		t.Skip("skipping on plan9; no DWARF symbol table in executables")
	}

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

	dir, err := ioutil.TempDir("", "TestEmbeddedStructMarker")
	if err != nil {
		t.Fatalf("could not create directory: %v", err)
	}
	defer os.RemoveAll(dir)

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
			name := entry.Val(dwarf.AttrName).(string)
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
	if runtime.GOOS == "plan9" {
		t.Skip("skipping on plan9; no DWARF symbol table in executables")
	}

	// External linking may bring in C symbols with unknown size. Skip.
	testenv.MustInternalLink(t)

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
	dir, err := ioutil.TempDir("", "TestSizes")
	if err != nil {
		t.Fatalf("could not create directory: %v", err)
	}
	defer os.RemoveAll(dir)
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
	if runtime.GOOS == "plan9" {
		t.Skip("skipping on plan9; no DWARF symbol table in executables")
	}
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
	dir, err := ioutil.TempDir("", "TestFieldOverlap")
	if err != nil {
		t.Fatalf("could not create directory: %v", err)
	}
	defer os.RemoveAll(dir)

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

func varDeclCoordsAndSubrogramDeclFile(t *testing.T, testpoint string, expectFile string, expectLine int, directive string) {
	t.Parallel()

	prog := fmt.Sprintf("package main\n%s\nfunc main() {\n\nvar i int\ni = i\n}\n", directive)

	dir, err := ioutil.TempDir("", testpoint)
	if err != nil {
		t.Fatalf("could not create directory: %v", err)
	}
	defer os.RemoveAll(dir)

	f := gobuild(t, dir, prog, NoOpt)

	d, err := f.DWARF()
	if err != nil {
		t.Fatalf("error reading DWARF: %v", err)
	}

	rdr := d.Reader()
	ex := examiner{}
	if err := ex.populate(rdr); err != nil {
		t.Fatalf("error reading DWARF: %v", err)
	}

	// Locate the main.main DIE
	mains := ex.Named("main.main")
	if len(mains) == 0 {
		t.Fatalf("unable to locate DIE for main.main")
	}
	if len(mains) != 1 {
		t.Fatalf("more than one main.main DIE")
	}
	maindie := mains[0]

	// Vet the main.main DIE
	if maindie.Tag != dwarf.TagSubprogram {
		t.Fatalf("unexpected tag %v on main.main DIE", maindie.Tag)
	}

	// Walk main's children and select variable "i".
	mainIdx := ex.idxFromOffset(maindie.Offset)
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
	line := iEntry.Val(dwarf.AttrDeclLine)
	if line == nil || line.(int64) != int64(expectLine) {
		t.Errorf("DW_AT_decl_line for i is %v, want %d", line, expectLine)
	}

	fileIdx, fileIdxOK := maindie.Val(dwarf.AttrDeclFile).(int64)
	if !fileIdxOK {
		t.Errorf("missing or invalid DW_AT_decl_file for main")
	}
	file := ex.FileRef(t, d, mainIdx, fileIdx)
	base := filepath.Base(file)
	if base != expectFile {
		t.Errorf("DW_AT_decl_file for main is %v, want %v", base, expectFile)
	}
}

func TestVarDeclCoordsAndSubrogramDeclFile(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	if runtime.GOOS == "plan9" {
		t.Skip("skipping on plan9; no DWARF symbol table in executables")
	}

	varDeclCoordsAndSubrogramDeclFile(t, "TestVarDeclCoords", "test.go", 5, "")
}

func TestVarDeclCoordsWithLineDirective(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	if runtime.GOOS == "plan9" {
		t.Skip("skipping on plan9; no DWARF symbol table in executables")
	}

	varDeclCoordsAndSubrogramDeclFile(t, "TestVarDeclCoordsWithLineDirective",
		"foobar.go", 202, "//line /foobar.go:200")
}

// Helper class for supporting queries on DIEs within a DWARF .debug_info
// section. Invoke the populate() method below passing in a dwarf.Reader,
// which will read in all DIEs and keep track of parent/child
// relationships. Queries can then be made to ask for DIEs by name or
// by offset. This will hopefully reduce boilerplate for future test
// writing.

type examiner struct {
	dies        []*dwarf.Entry
	idxByOffset map[dwarf.Offset]int
	kids        map[int][]int
	parent      map[int]int
	byname      map[string][]int
}

// Populate the examiner using the DIEs read from rdr.
func (ex *examiner) populate(rdr *dwarf.Reader) error {
	ex.idxByOffset = make(map[dwarf.Offset]int)
	ex.kids = make(map[int][]int)
	ex.parent = make(map[int]int)
	ex.byname = make(map[string][]int)
	var nesting []int
	for entry, err := rdr.Next(); entry != nil; entry, err = rdr.Next() {
		if err != nil {
			return err
		}
		if entry.Tag == 0 {
			// terminator
			if len(nesting) == 0 {
				return errors.New("nesting stack underflow")
			}
			nesting = nesting[:len(nesting)-1]
			continue
		}
		idx := len(ex.dies)
		ex.dies = append(ex.dies, entry)
		if _, found := ex.idxByOffset[entry.Offset]; found {
			return errors.New("DIE clash on offset")
		}
		ex.idxByOffset[entry.Offset] = idx
		if name, ok := entry.Val(dwarf.AttrName).(string); ok {
			ex.byname[name] = append(ex.byname[name], idx)
		}
		if len(nesting) > 0 {
			parent := nesting[len(nesting)-1]
			ex.kids[parent] = append(ex.kids[parent], idx)
			ex.parent[idx] = parent
		}
		if entry.Children {
			nesting = append(nesting, idx)
		}
	}
	if len(nesting) > 0 {
		return errors.New("unterminated child sequence")
	}
	return nil
}

func indent(ilevel int) {
	for i := 0; i < ilevel; i++ {
		fmt.Printf("  ")
	}
}

// For debugging new tests
func (ex *examiner) dumpEntry(idx int, dumpKids bool, ilevel int) error {
	if idx >= len(ex.dies) {
		msg := fmt.Sprintf("bad DIE %d: index out of range\n", idx)
		return errors.New(msg)
	}
	entry := ex.dies[idx]
	indent(ilevel)
	fmt.Printf("0x%x: %v\n", idx, entry.Tag)
	for _, f := range entry.Field {
		indent(ilevel)
		fmt.Printf("at=%v val=0x%x\n", f.Attr, f.Val)
	}
	if dumpKids {
		ksl := ex.kids[idx]
		for _, k := range ksl {
			ex.dumpEntry(k, true, ilevel+2)
		}
	}
	return nil
}

// Given a DIE offset, return the previously read dwarf.Entry, or nil
func (ex *examiner) entryFromOffset(off dwarf.Offset) *dwarf.Entry {
	if idx, found := ex.idxByOffset[off]; found && idx != -1 {
		return ex.entryFromIdx(idx)
	}
	return nil
}

// Return the ID that examiner uses to refer to the DIE at offset off
func (ex *examiner) idxFromOffset(off dwarf.Offset) int {
	if idx, found := ex.idxByOffset[off]; found {
		return idx
	}
	return -1
}

// Return the dwarf.Entry pointer for the DIE with id 'idx'
func (ex *examiner) entryFromIdx(idx int) *dwarf.Entry {
	if idx >= len(ex.dies) || idx < 0 {
		return nil
	}
	return ex.dies[idx]
}

// Returns a list of child entries for a die with ID 'idx'
func (ex *examiner) Children(idx int) []*dwarf.Entry {
	sl := ex.kids[idx]
	ret := make([]*dwarf.Entry, len(sl))
	for i, k := range sl {
		ret[i] = ex.entryFromIdx(k)
	}
	return ret
}

// Returns parent DIE for DIE 'idx', or nil if the DIE is top level
func (ex *examiner) Parent(idx int) *dwarf.Entry {
	p, found := ex.parent[idx]
	if !found {
		return nil
	}
	return ex.entryFromIdx(p)
}

// ParentCU returns the enclosing compilation unit DIE for the DIE
// with a given index, or nil if for some reason we can't establish a
// parent.
func (ex *examiner) ParentCU(idx int) *dwarf.Entry {
	for {
		parentDie := ex.Parent(idx)
		if parentDie == nil {
			return nil
		}
		if parentDie.Tag == dwarf.TagCompileUnit {
			return parentDie
		}
		idx = ex.idxFromOffset(parentDie.Offset)
	}
}

// FileRef takes a given DIE by index and a numeric file reference
// (presumably from a decl_file or call_file attribute), looks up the
// reference in the .debug_line file table, and returns the proper
// string for it. We need to know which DIE is making the reference
// so as find the right compilation unit.
func (ex *examiner) FileRef(t *testing.T, dw *dwarf.Data, dieIdx int, fileRef int64) string {

	// Find the parent compilation unit DIE for the specified DIE.
	cuDie := ex.ParentCU(dieIdx)
	if cuDie == nil {
		t.Fatalf("no parent CU DIE for DIE with idx %d?", dieIdx)
		return ""
	}
	// Construct a line reader and then use it to get the file string.
	lr, lrerr := dw.LineReader(cuDie)
	if lrerr != nil {
		t.Fatal("d.LineReader: ", lrerr)
		return ""
	}
	files := lr.Files()
	if fileRef < 0 || int(fileRef) > len(files)-1 {
		t.Fatalf("examiner.FileRef: malformed file reference %d", fileRef)
		return ""
	}
	return files[fileRef].Name
}

// Return a list of all DIEs with name 'name'. When searching for DIEs
// by name, keep in mind that the returned results will include child
// DIEs such as params/variables. For example, asking for all DIEs named
// "p" for even a small program will give you 400-500 entries.
func (ex *examiner) Named(name string) []*dwarf.Entry {
	sl := ex.byname[name]
	ret := make([]*dwarf.Entry, len(sl))
	for i, k := range sl {
		ret[i] = ex.entryFromIdx(k)
	}
	return ret
}

func TestInlinedRoutineRecords(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	if runtime.GOOS == "plan9" {
		t.Skip("skipping on plan9; no DWARF symbol table in executables")
	}
	if runtime.GOOS == "solaris" || runtime.GOOS == "illumos" || runtime.GOOS == "darwin" {
		t.Skip("skipping on solaris, illumos, and darwin, pending resolution of issue #23168")
	}

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
	dir, err := ioutil.TempDir("", "TestInlinedRoutineRecords")
	if err != nil {
		t.Fatalf("could not create directory: %v", err)
	}
	defer os.RemoveAll(dir)

	// Note: this is a build with "-l=4", as opposed to "-l -N". The
	// test is intended to verify DWARF that is only generated when
	// the inliner is active. We're only going to look at the DWARF for
	// main.main, however, hence we build with "-gcflags=-l=4" as opposed
	// to "-gcflags=all=-l=4".
	f := gobuild(t, dir, prog, OptInl4)

	d, err := f.DWARF()
	if err != nil {
		t.Fatalf("error reading DWARF: %v", err)
	}

	// The inlined subroutines we expect to visit
	expectedInl := []string{"main.cand"}

	rdr := d.Reader()
	ex := examiner{}
	if err := ex.populate(rdr); err != nil {
		t.Fatalf("error reading DWARF: %v", err)
	}

	// Locate the main.main DIE
	mains := ex.Named("main.main")
	if len(mains) == 0 {
		t.Fatalf("unable to locate DIE for main.main")
	}
	if len(mains) != 1 {
		t.Fatalf("more than one main.main DIE")
	}
	maindie := mains[0]

	// Vet the main.main DIE
	if maindie.Tag != dwarf.TagSubprogram {
		t.Fatalf("unexpected tag %v on main.main DIE", maindie.Tag)
	}

	// Walk main's children and pick out the inlined subroutines
	mainIdx := ex.idxFromOffset(maindie.Offset)
	childDies := ex.Children(mainIdx)
	exCount := 0
	for _, child := range childDies {
		if child.Tag == dwarf.TagInlinedSubroutine {
			// Found an inlined subroutine, locate abstract origin.
			ooff, originOK := child.Val(dwarf.AttrAbstractOrigin).(dwarf.Offset)
			if !originOK {
				t.Fatalf("no abstract origin attr for inlined subroutine at offset %v", child.Offset)
			}
			originDIE := ex.entryFromOffset(ooff)
			if originDIE == nil {
				t.Fatalf("can't locate origin DIE at off %v", ooff)
			}

			// Walk the children of the abstract subroutine. We expect
			// to see child variables there, even if (perhaps due to
			// optimization) there are no references to them from the
			// inlined subroutine DIE.
			absFcnIdx := ex.idxFromOffset(ooff)
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

			if exCount >= len(expectedInl) {
				t.Fatalf("too many inlined subroutines found in main.main")
			}

			// Name should check out.
			expected := expectedInl[exCount]
			if name, ok := originDIE.Val(dwarf.AttrName).(string); ok {
				if name != expected {
					t.Fatalf("expected inlined routine %s got %s", name, expected)
				}
			}
			exCount++

			// Verify that the call_file attribute for the inlined
			// instance is ok. In this case it should match the file
			// for the main routine. To do this we need to locate the
			// compilation unit DIE that encloses what we're looking
			// at; this can be done with the examiner.
			cf, cfOK := child.Val(dwarf.AttrCallFile).(int64)
			if !cfOK {
				t.Fatalf("no call_file attr for inlined subroutine at offset %v", child.Offset)
			}
			file := ex.FileRef(t, d, mainIdx, cf)
			base := filepath.Base(file)
			if base != "test.go" {
				t.Errorf("bad call_file attribute, found '%s', want '%s'",
					file, "test.go")
			}

			omap := make(map[dwarf.Offset]bool)

			// Walk the child variables of the inlined routine. Each
			// of them should have a distinct abstract origin-- if two
			// vars point to the same origin things are definitely broken.
			inlIdx := ex.idxFromOffset(child.Offset)
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
	}
	if exCount != len(expectedInl) {
		t.Fatalf("not enough inlined subroutines found in main.main")
	}
}

func abstractOriginSanity(t *testing.T, pkgDir string, flags string) {
	t.Parallel()

	dir, err := ioutil.TempDir("", "TestAbstractOriginSanity")
	if err != nil {
		t.Fatalf("could not create directory: %v", err)
	}
	defer os.RemoveAll(dir)

	// Build with inlining, to exercise DWARF inlining support.
	f := gobuildTestdata(t, dir, filepath.Join(pkgDir, "main"), flags)

	d, err := f.DWARF()
	if err != nil {
		t.Fatalf("error reading DWARF: %v", err)
	}
	rdr := d.Reader()
	ex := examiner{}
	if err := ex.populate(rdr); err != nil {
		t.Fatalf("error reading DWARF: %v", err)
	}

	// Make a pass through all DIEs looking for abstract origin
	// references.
	abscount := 0
	for i, die := range ex.dies {
		// Does it have an abstract origin?
		ooff, originOK := die.Val(dwarf.AttrAbstractOrigin).(dwarf.Offset)
		if !originOK {
			continue
		}

		// All abstract origin references should be resolvable.
		abscount += 1
		originDIE := ex.entryFromOffset(ooff)
		if originDIE == nil {
			ex.dumpEntry(i, false, 0)
			t.Fatalf("unresolved abstract origin ref in DIE at offset 0x%x\n", die.Offset)
		}

		// Suppose that DIE X has parameter/variable children {K1,
		// K2, ... KN}. If X has an abstract origin of A, then for
		// each KJ, the abstract origin of KJ should be a child of A.
		// Note that this same rule doesn't hold for non-variable DIEs.
		pidx := ex.idxFromOffset(die.Offset)
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
			childOriginDIE := ex.entryFromOffset(kooff)
			if childOriginDIE == nil {
				ex.dumpEntry(i, false, 0)
				t.Fatalf("unresolved abstract origin ref in DIE at offset %x", kid.Offset)
			}
			coidx := ex.idxFromOffset(childOriginDIE.Offset)
			childOriginParent := ex.Parent(coidx)
			if childOriginParent != originDIE {
				ex.dumpEntry(i, false, 0)
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

	if runtime.GOOS == "plan9" {
		t.Skip("skipping on plan9; no DWARF symbol table in executables")
	}
	if runtime.GOOS == "solaris" || runtime.GOOS == "illumos" || runtime.GOOS == "darwin" {
		t.Skip("skipping on solaris, illumos, and darwin, pending resolution of issue #23168")
	}

	if wd, err := os.Getwd(); err == nil {
		gopathdir := filepath.Join(wd, "testdata", "httptest")
		abstractOriginSanity(t, gopathdir, OptAllInl4)
	} else {
		t.Fatalf("os.Getwd() failed %v", err)
	}
}

func TestAbstractOriginSanityIssue25459(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	if runtime.GOOS == "plan9" {
		t.Skip("skipping on plan9; no DWARF symbol table in executables")
	}
	if runtime.GOOS == "solaris" || runtime.GOOS == "illumos" || runtime.GOOS == "darwin" {
		t.Skip("skipping on solaris, illumos, and darwin, pending resolution of issue #23168")
	}
	if runtime.GOARCH != "amd64" && runtime.GOARCH != "x86" {
		t.Skip("skipping on not-amd64 not-x86; location lists not supported")
	}

	if wd, err := os.Getwd(); err == nil {
		gopathdir := filepath.Join(wd, "testdata", "issue25459")
		abstractOriginSanity(t, gopathdir, DefaultOpt)
	} else {
		t.Fatalf("os.Getwd() failed %v", err)
	}
}

func TestAbstractOriginSanityIssue26237(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	if runtime.GOOS == "plan9" {
		t.Skip("skipping on plan9; no DWARF symbol table in executables")
	}
	if runtime.GOOS == "solaris" || runtime.GOOS == "illumos" || runtime.GOOS == "darwin" {
		t.Skip("skipping on solaris, illumos, and darwin, pending resolution of issue #23168")
	}
	if wd, err := os.Getwd(); err == nil {
		gopathdir := filepath.Join(wd, "testdata", "issue26237")
		abstractOriginSanity(t, gopathdir, DefaultOpt)
	} else {
		t.Fatalf("os.Getwd() failed %v", err)
	}
}

func TestRuntimeTypeAttrInternal(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	testenv.MustInternalLink(t)

	if runtime.GOOS == "plan9" {
		t.Skip("skipping on plan9; no DWARF symbol table in executables")
	}

	if runtime.GOOS == "windows" {
		t.Skip("skipping on windows; test is incompatible with relocatable binaries")
	}

	testRuntimeTypeAttr(t, "-ldflags=-linkmode=internal")
}

// External linking requires a host linker (https://golang.org/src/cmd/cgo/doc.go l.732)
func TestRuntimeTypeAttrExternal(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	testenv.MustHaveCGO(t)

	if runtime.GOOS == "plan9" {
		t.Skip("skipping on plan9; no DWARF symbol table in executables")
	}

	// Explicitly test external linking, for dsymutil compatibility on Darwin.
	if runtime.GOARCH == "ppc64" {
		t.Skip("-linkmode=external not supported on ppc64")
	}

	if runtime.GOOS == "windows" {
		t.Skip("skipping on windows; test is incompatible with relocatable binaries")
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
	dir, err := ioutil.TempDir("", "TestRuntimeType")
	if err != nil {
		t.Fatalf("could not create directory: %v", err)
	}
	defer os.RemoveAll(dir)

	f := gobuild(t, dir, prog, flags)
	out, err := exec.Command(f.path).CombinedOutput()
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
	ex := examiner{}
	if err := ex.populate(rdr); err != nil {
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

	if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" {
		return // everything is PIE on ARM64, addresses are relocated
	}
	if rtAttr.(uint64)+types.Addr != addr {
		t.Errorf("DWARF type offset was %#x+%#x, but test program said %#x", rtAttr.(uint64), types.Addr, addr)
	}
}

func TestIssue27614(t *testing.T) {
	// Type references in debug_info should always use the DW_TAG_typedef_type
	// for the type, when that's generated.

	testenv.MustHaveGoBuild(t)

	if runtime.GOOS == "plan9" {
		t.Skip("skipping on plan9; no DWARF symbol table in executables")
	}

	t.Parallel()

	dir, err := ioutil.TempDir("", "go-build")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)

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

	if runtime.GOOS == "plan9" {
		t.Skip("skipping on plan9; no DWARF symbol table in executables")
	}

	t.Parallel()

	dir, err := ioutil.TempDir("", "go-build")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)

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
	if !testenv.CanInternalLink() {
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

	if runtime.GOOS == "plan9" {
		t.Skip("skipping on plan9; no DWARF symbol table in executables")
	}

	t.Parallel()

	dir, err := ioutil.TempDir("", "go-build")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)

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

	tmpdir, err := ioutil.TempDir("", "TestMachoIssue32233")
	if err != nil {
		t.Fatalf("could not create directory: %v", err)
	}
	defer os.RemoveAll(tmpdir)

	wd, err2 := os.Getwd()
	if err2 != nil {
		t.Fatalf("where am I? %v", err)
	}
	pdir := filepath.Join(wd, "testdata", "issue32233", "main")
	f := gobuildTestdata(t, tmpdir, pdir, DefaultOpt)
	f.Close()
}

func TestWindowsIssue36495(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	if runtime.GOOS != "windows" {
		t.Skip("skipping: test only on windows")
	}

	dir, err := ioutil.TempDir("", "TestEmbeddedStructMarker")
	if err != nil {
		t.Fatalf("could not create directory: %v", err)
	}
	defer os.RemoveAll(dir)

	prog := `
package main

import "fmt"

func main() {
  fmt.Println("Hello World")
}`
	f := gobuild(t, dir, prog, NoOpt)
	exe, err := pe.Open(f.path)
	if err != nil {
		t.Fatalf("error opening pe file: %v", err)
	}
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

	if runtime.GOOS == "plan9" {
		t.Skip("skipping on plan9; no DWARF symbol table in executables")
	}

	t.Parallel()

	// Build a test program that contains a translation unit whose
	// text (from am assembly source) contains only a single instruction.
	tmpdir, err := ioutil.TempDir("", "TestIssue38192")
	if err != nil {
		t.Fatalf("could not create directory: %v", err)
	}
	defer os.RemoveAll(tmpdir)
	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("where am I? %v", err)
	}
	pdir := filepath.Join(wd, "testdata", "issue38192")
	f := gobuildTestdata(t, tmpdir, pdir, DefaultOpt)

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

	if runtime.GOOS == "plan9" {
		t.Skip("skipping on plan9; no DWARF symbol table in executables")
	}

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

	tmpdir, err := ioutil.TempDir("", "TestIssue38192")
	if err != nil {
		t.Fatalf("could not create directory: %v", err)
	}
	defer os.RemoveAll(tmpdir)
	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("where am I? %v", err)
	}
	pdir := filepath.Join(wd, "testdata", "issue39757")
	f := gobuildTestdata(t, tmpdir, pdir, DefaultOpt)

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
	ex := examiner{}
	if err := ex.populate(rdr); err != nil {
		t.Fatalf("error reading DWARF: %v", err)
	}

	// Locate the main.main DIE
	mains := ex.Named("main.main")
	if len(mains) == 0 {
		t.Fatalf("unable to locate DIE for main.main")
	}
	if len(mains) != 1 {
		t.Fatalf("more than one main.main DIE")
	}
	maindie := mains[0]

	// Collect the start/end PC for main.main
	lowpc := maindie.Val(dwarf.AttrLowpc).(uint64)
	highpc := maindie.Val(dwarf.AttrHighpc).(uint64)

	// Now read the line table for the 'main' compilation unit.
	mainIdx := ex.idxFromOffset(maindie.Offset)
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
