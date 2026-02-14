package gosym

import "unique"

func (t *Table) funcName(off uint32) unique.Handle[string] {
	offset := t.funcnametab[0] + uint64(off)
	if offset >= t.funcnametab[1] {
		return unique.Handle[string]{}
	}
	end := offset
	for end < t.funcnametab[1] && t.pclntab[end] != 0 {
		end++
	}
	return unique.Make(string(t.pclntab[offset:end]))
}

func (t *Table) fileName(off uint32) unique.Handle[string] {
	offset := t.filetab[0] + uint64(off)
	if offset >= t.filetab[1] {
		return unique.Handle[string]{}
	}
	end := offset
	for end < t.filetab[1] && t.pclntab[end] != 0 {
		end++
	}
	return unique.Make(string(t.pclntab[offset:end]))
}
