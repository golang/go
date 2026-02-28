// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dwtest

import (
	"debug/dwarf"
	"errors"
	"fmt"
	"os"
)

// Helper type for supporting queries on DIEs within a DWARF
// .debug_info section. Invoke the populate() method below passing in
// a dwarf.Reader, which will read in all DIEs and keep track of
// parent/child relationships. Queries can then be made to ask for
// DIEs by name or by offset. This will hopefully reduce boilerplate
// for future test writing.

type Examiner struct {
	dies        []*dwarf.Entry
	idxByOffset map[dwarf.Offset]int
	kids        map[int][]int
	parent      map[int]int
	byname      map[string][]int
}

// Populate the Examiner using the DIEs read from rdr.
func (ex *Examiner) Populate(rdr *dwarf.Reader) error {
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

func (e *Examiner) DIEs() []*dwarf.Entry {
	return e.dies
}

func indent(ilevel int) {
	for i := 0; i < ilevel; i++ {
		fmt.Printf("  ")
	}
}

// For debugging new tests
func (ex *Examiner) DumpEntry(idx int, dumpKids bool, ilevel int) {
	if idx >= len(ex.dies) {
		fmt.Fprintf(os.Stderr, "DumpEntry: bad DIE %d: index out of range\n", idx)
		return
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
			ex.DumpEntry(k, true, ilevel+2)
		}
	}
}

// Given a DIE offset, return the previously read dwarf.Entry, or nil
func (ex *Examiner) EntryFromOffset(off dwarf.Offset) *dwarf.Entry {
	if idx, found := ex.idxByOffset[off]; found && idx != -1 {
		return ex.entryFromIdx(idx)
	}
	return nil
}

// Return the ID that Examiner uses to refer to the DIE at offset off
func (ex *Examiner) IdxFromOffset(off dwarf.Offset) int {
	if idx, found := ex.idxByOffset[off]; found {
		return idx
	}
	return -1
}

// Return the dwarf.Entry pointer for the DIE with id 'idx'
func (ex *Examiner) entryFromIdx(idx int) *dwarf.Entry {
	if idx >= len(ex.dies) || idx < 0 {
		return nil
	}
	return ex.dies[idx]
}

// Returns a list of child entries for a die with ID 'idx'
func (ex *Examiner) Children(idx int) []*dwarf.Entry {
	sl := ex.kids[idx]
	ret := make([]*dwarf.Entry, len(sl))
	for i, k := range sl {
		ret[i] = ex.entryFromIdx(k)
	}
	return ret
}

// Returns parent DIE for DIE 'idx', or nil if the DIE is top level
func (ex *Examiner) Parent(idx int) *dwarf.Entry {
	p, found := ex.parent[idx]
	if !found {
		return nil
	}
	return ex.entryFromIdx(p)
}

// ParentCU returns the enclosing compilation unit DIE for the DIE
// with a given index, or nil if for some reason we can't establish a
// parent.
func (ex *Examiner) ParentCU(idx int) *dwarf.Entry {
	for {
		parentDie := ex.Parent(idx)
		if parentDie == nil {
			return nil
		}
		if parentDie.Tag == dwarf.TagCompileUnit {
			return parentDie
		}
		idx = ex.IdxFromOffset(parentDie.Offset)
	}
}

// FileRef takes a given DIE by index and a numeric file reference
// (presumably from a decl_file or call_file attribute), looks up the
// reference in the .debug_line file table, and returns the proper
// string for it. We need to know which DIE is making the reference
// so as to find the right compilation unit.
func (ex *Examiner) FileRef(dw *dwarf.Data, dieIdx int, fileRef int64) (string, error) {

	// Find the parent compilation unit DIE for the specified DIE.
	cuDie := ex.ParentCU(dieIdx)
	if cuDie == nil {
		return "", fmt.Errorf("no parent CU DIE for DIE with idx %d?", dieIdx)
	}
	// Construct a line reader and then use it to get the file string.
	lr, lrerr := dw.LineReader(cuDie)
	if lrerr != nil {
		return "", fmt.Errorf("d.LineReader: %v", lrerr)
	}
	files := lr.Files()
	if fileRef < 0 || int(fileRef) > len(files)-1 {
		return "", fmt.Errorf("Examiner.FileRef: malformed file reference %d", fileRef)
	}
	return files[fileRef].Name, nil
}

// Return a list of all DIEs with name 'name'. When searching for DIEs
// by name, keep in mind that the returned results will include child
// DIEs such as params/variables. For example, asking for all DIEs named
// "p" for even a small program will give you 400-500 entries.
func (ex *Examiner) Named(name string) []*dwarf.Entry {
	sl := ex.byname[name]
	ret := make([]*dwarf.Entry, len(sl))
	for i, k := range sl {
		ret[i] = ex.entryFromIdx(k)
	}
	return ret
}
