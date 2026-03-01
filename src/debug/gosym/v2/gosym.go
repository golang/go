// Package gosym implements access to the Go symbol tables embedded in Go binaries.
package gosym

import (
	"debug/elf"
	"debug/macho"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"iter"
	"unique"
)

var (
	// ErrPcNotFound is returned when:
	//  - PC doesn't belong to any function in case of table search
	//  - PC doesn't belong to the given function in case of function search
	ErrPcNotFound = errors.New("pc not found")
	// ErrCorrupted is returned in wide range of cases of binary data inconsistencies.
	ErrCorrupted = errors.New("binary data is corrupted")
)

// Table represents the Go symbols table.
type Table struct {
	metadata
	pclntab []byte
	gofunc  []byte
}

// NewELF creates a new table from an ELF file.
func NewELF(elf *elf.File) (*Table, error) {
	return parseObject(elfObject{elf})
}

// NewMacho creates a new table from a Mach-O file.
func NewMacho(mach *macho.File) (*Table, error) {
	return parseObject(machObject{mach})
}

// TODO: support PE and Plan9 objects
// func NewPE(*pe.File) (*Table, error)
// func NewPlan9Obj(*plan9obj.File) (*Table, error)

// NewMagic creates a new table from an object file, auto-detecting amongs the supported formats.
func NewMagic(r io.ReaderAt) (*Table, error) {
	return newMagic(r)
}

// NewObject creates a new table from an abstract representation of an object file.
func NewObject(obj Object) (*Table, error) {
	return parseObject(obj)
}

// Object represents an object file.
type Object interface {
	Endian() binary.ByteOrder
	Sections() ([]SectionHeader, error)
	SectionData(i int8) ([]byte, error)
}

// SectionHeader represents a header of a section in an object file.
type SectionHeader struct {
	Name string
	Addr uint64
	Size uint64
}

// ResolveLocations resolves source code locations associated with a given PC.
// Returns more than one item only in case of inline functions, which are
// returned in inner to outer-most inlining order.
func (t *Table) ResolveLocations(pc uint64, buf []Location) ([]Location, error) {
	f, err := t.ResolveFunction(pc)
	if err != nil {
		return buf, err
	}
	return f.resolveLocations(pc, buf)
}

// Location represents a source code location.
type Location struct {
	Function unique.Handle[string]
	File     unique.Handle[string]
	Line     uint32
}

// Functions lists all functions in the table. Note that some functions
// may be inlined by the compiler, and not appear directly in the table.
// These are accessible by listing inline functions.
func (t *Table) Functions() iter.Seq2[Function, error] {
	return func(yield func(Function, error) bool) {
		for i := uint32(0); i < t.functabCount(); i++ {
			off, err := t.functabOff(i)
			if err != nil {
				if !yield(Function{}, err) {
					return
				}
			}
			cuOff, err := t.funcdataCuOffset(off)
			if err != nil {
				if !yield(Function{}, err) {
					return
				}
			}
			if cuOff == ^uint32(0) {
				// Non-function entry.
				continue
			}
			if !yield(Function{
				table:  t,
				idx:    i,
				offset: off,
			}, nil) {
				return
			}
		}
	}
}

// ResolveFunction resolves function symbol that the given pc corresponds to.
func (t *Table) ResolveFunction(pc uint64) (Function, error) {
	idx, err := t.functabIdxByPc(pc)
	if err != nil {
		return Function{}, err
	}
	off, err := t.functabOff(idx)
	if err != nil {
		return Function{}, err
	}
	cuOff, err := t.funcdataCuOffset(off)
	if err != nil {
		return Function{}, err
	}
	if cuOff == ^uint32(0) {
		return Function{}, fmt.Errorf("%w: pc not in a function", ErrPcNotFound)
	}
	f := Function{
		table:  t,
		idx:    idx,
		offset: off,
	}
	end, err := f.End()
	if err != nil {
		return Function{}, err
	}
	if pc >= end {
		return Function{}, fmt.Errorf("%w: pc not in a function", ErrPcNotFound)
	}
	return f, nil
}

// Function represents a function entry in the table.
type Function struct {
	table  *Table
	idx    uint32
	offset uint64
}

// Name returns the name of the function.
func (f *Function) Name() (unique.Handle[string], error) {
	nameOff, err := f.table.funcdataNameOff(f.offset)
	if err != nil {
		return unique.Handle[string]{}, err
	}
	return f.table.funcName(nameOff), nil
}

// Entry returns the lowest PC of the function.
func (f *Function) Entry() (uint64, error) {
	return f.table.functabPc(f.idx)
}

// End returns the PC past the end of the function.
func (f *Function) End() (uint64, error) {
	return f.endPc()
}

// DeferReturn returns the deferreturn address if any, 0 otherwise.
func (f *Function) DeferReturn() (uint32, error) {
	return f.table.funcdataDeferReturn(f.offset)
}

// File returns the file that the function is defined in.
func (f *Function) File() (unique.Handle[string], error) {
	off, err := f.table.funcdataFileOff(f.offset)
	if err != nil {
		return unique.Handle[string]{}, err
	}
	return f.table.fileName(off), nil
}

// StartLine returns the line number the function is defined at.
func (f *Function) StartLine() (uint32, error) {
	return f.table.funcdataStartLine(f.offset)
}

// ResolveLocations resolves source code locations associated with a given PC
// that belongs to the function. Returns more than one item only in case of
// inline functions, which are returned in inner to outer-most inlining order.
func (f *Function) ResolveLocations(pc uint64, buf []Location) ([]Location, error) {
	return f.resolveLocations(pc, buf)
}

// InlineFunctions lists all functions that have been inlined in this function (directly or indirectly).
func (f *Function) InlineFunctions(buf []InlineFunction) ([]InlineFunction, error) {
	return f.inlines(buf)
}

// InlineFunction represents a function that has been inlined.
type InlineFunction struct {
	Name      unique.Handle[string]
	File      unique.Handle[string]
	StartLine uint32
}

// Lines resolves all source code locations of the function.
func (f *Function) Lines(buf LinesResult) (LinesResult, error) {
	return f.lines(buf)
}

// LinesResult represents the result of function lines resolution. The
// object may be reused for any subsequent calls to optimize allocations.
// FunctionLines slice must be copied if access to its elements is needed
// after the result object is re-used for subsequent calls.
type LinesResult struct {
	FunctionLines []FunctionLine
	linesCache    linesCache
}

// FunctionLine represents a PC range that correpsonds to a specific source code location.
type FunctionLine struct {
	PCLo     uint64
	PCHi     uint64
	Name     unique.Handle[string]
	File     unique.Handle[string]
	Line     uint32
	ParentPC uint64
}
