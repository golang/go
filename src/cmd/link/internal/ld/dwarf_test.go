// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	objfilepkg "cmd/internal/objfile" // renamed to avoid conflict with objfile function
	"debug/dwarf"
	"errors"
	"fmt"
	"internal/testenv"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"testing"
)

const (
	NoOpt        = "-gcflags=-l -N"
	Opt          = ""
	OptInl4      = "-gcflags=all=-l=4"
	OptInl4DwLoc = "-gcflags=all=-l=4 -dwarflocationlists"
)

func TestRuntimeTypeDIEs(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	if runtime.GOOS == "plan9" {
		t.Skip("skipping on plan9; no DWARF symbol table in executables")
	}

	dir, err := ioutil.TempDir("", "TestRuntimeTypeDIEs")
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

func gobuild(t *testing.T, dir string, testfile string, gcflags string) *objfilepkg.File {
	src := filepath.Join(dir, "test.go")
	dst := filepath.Join(dir, "out")

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
	return f
}

func TestEmbeddedStructMarker(t *testing.T) {
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
		"main.Foo": map[string]bool{"v": false},
		"main.Bar": map[string]bool{"Foo": true, "name": false},
		"main.Baz": map[string]bool{"Foo": true, "name": false},
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
	const goEmbeddedStruct = dwarf.Attr(0x2903)
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

func TestVarDeclCoordsAndSubrogramDeclFile(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	if runtime.GOOS == "plan9" {
		t.Skip("skipping on plan9; no DWARF symbol table in executables")
	}

	const prog = `
package main

func main() {
	var i int
	i = i
}
`
	dir, err := ioutil.TempDir("", "TestVarDeclCoords")
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
	if line == nil || line.(int64) != 5 {
		t.Errorf("DW_AT_decl_line for i is %v, want 5", line)
	}

	file := maindie.Val(dwarf.AttrDeclFile)
	if file == nil || file.(int64) != 1 {
		t.Errorf("DW_AT_decl_file for main is %v, want 1", file)
	}
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

// Return the ID that that examiner uses to refer to the DIE at offset off
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
	if runtime.GOOS == "solaris" {
		t.Skip("skipping on solaris, pending resolution of issue #23168")
	}

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
	// the inliner is active.
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

func abstractOriginSanity(t *testing.T, flags string) {

	// Nothing special about net/http here, this is just a convenient
	// way to pull in a lot of code.
	const prog = `
package main

import (
	"net/http"
	"net/http/httptest"
)

type statusHandler int

func (h *statusHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(int(*h))
}

func main() {
	status := statusHandler(http.StatusNotFound)
	s := httptest.NewServer(&status)
	defer s.Close()
}
`
	dir, err := ioutil.TempDir("", "TestAbstractOriginSanity")
	if err != nil {
		t.Fatalf("could not create directory: %v", err)
	}
	defer os.RemoveAll(dir)

	// Build with inlining, to exercise DWARF inlining support.
	f := gobuild(t, dir, prog, flags)

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

	if runtime.GOOS == "plan9" {
		t.Skip("skipping on plan9; no DWARF symbol table in executables")
	}
	if runtime.GOOS == "solaris" {
		t.Skip("skipping on solaris, pending resolution of issue #23168")
	}

	abstractOriginSanity(t, OptInl4)
}

func TestAbstractOriginSanityWithLocationLists(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	if runtime.GOOS == "plan9" {
		t.Skip("skipping on plan9; no DWARF symbol table in executables")
	}
	if runtime.GOOS == "solaris" {
		t.Skip("skipping on solaris, pending resolution of issue #23168")
	}
	if runtime.GOARCH != "amd64" && runtime.GOARCH != "x86" {
		t.Skip("skipping on not-amd64 not-x86; location lists not supported")
	}

	abstractOriginSanity(t, OptInl4DwLoc)
}
