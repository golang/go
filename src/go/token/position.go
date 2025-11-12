// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package token

import (
	"cmp"
	"fmt"
	"slices"
	"strconv"
	"sync"
	"sync/atomic"
)

// If debug is set, invalid offset and position values cause a panic
// (go.dev/issue/57490).
const debug = false

// -----------------------------------------------------------------------------
// Positions

// Position describes an arbitrary source position
// including the file, line, and column location.
// A Position is valid if the line number is > 0.
type Position struct {
	Filename string // filename, if any
	Offset   int    // offset, starting at 0
	Line     int    // line number, starting at 1
	Column   int    // column number, starting at 1 (byte count)
}

// IsValid reports whether the position is valid.
func (pos *Position) IsValid() bool { return pos.Line > 0 }

// String returns a string in one of several forms:
//
//	file:line:column    valid position with file name
//	file:line           valid position with file name but no column (column == 0)
//	line:column         valid position without file name
//	line                valid position without file name and no column (column == 0)
//	file                invalid position with file name
//	-                   invalid position without file name
func (pos Position) String() string {
	s := pos.Filename
	if pos.IsValid() {
		if s != "" {
			s += ":"
		}
		s += strconv.Itoa(pos.Line)
		if pos.Column != 0 {
			s += fmt.Sprintf(":%d", pos.Column)
		}
	}
	if s == "" {
		s = "-"
	}
	return s
}

// Pos is a compact encoding of a source position within a file set.
// It can be converted into a [Position] for a more convenient, but much
// larger, representation.
//
// The Pos value for a given file is a number in the range [base, base+size],
// where base and size are specified when a file is added to the file set.
// The difference between a Pos value and the corresponding file base
// corresponds to the byte offset of that position (represented by the Pos value)
// from the beginning of the file. Thus, the file base offset is the Pos value
// representing the first byte in the file.
//
// To create the Pos value for a specific source offset (measured in bytes),
// first add the respective file to the current file set using [FileSet.AddFile]
// and then call [File.Pos](offset) for that file. Given a Pos value p
// for a specific file set fset, the corresponding [Position] value is
// obtained by calling fset.Position(p).
//
// Pos values can be compared directly with the usual comparison operators:
// If two Pos values p and q are in the same file, comparing p and q is
// equivalent to comparing the respective source file offsets. If p and q
// are in different files, p < q is true if the file implied by p was added
// to the respective file set before the file implied by q.
type Pos int

// The zero value for [Pos] is NoPos; there is no file and line information
// associated with it, and NoPos.IsValid() is false. NoPos is always
// smaller than any other [Pos] value. The corresponding [Position] value
// for NoPos is the zero value for [Position].
const NoPos Pos = 0

// IsValid reports whether the position is valid.
func (p Pos) IsValid() bool {
	return p != NoPos
}

// -----------------------------------------------------------------------------
// File

// A File is a handle for a file belonging to a [FileSet].
// A File has a name, size, and line offset table.
//
// Use [FileSet.AddFile] to create a File.
// A File may belong to more than one FileSet; see [FileSet.AddExistingFiles].
type File struct {
	name string // file name as provided to AddFile
	base int    // Pos value range for this file is [base...base+size]
	size int    // file size as provided to AddFile

	// lines and infos are protected by mutex
	mutex sync.Mutex
	lines []int // lines contains the offset of the first character for each line (the first entry is always 0)
	infos []lineInfo
}

// Name returns the file name of file f as registered with AddFile.
func (f *File) Name() string {
	return f.name
}

// Base returns the base offset of file f as registered with AddFile.
func (f *File) Base() int {
	return f.base
}

// Size returns the size of file f as registered with AddFile.
func (f *File) Size() int {
	return f.size
}

// End returns the end position of file f as registered with AddFile.
func (f *File) End() Pos {
	return Pos(f.base + f.size)
}

// LineCount returns the number of lines in file f.
func (f *File) LineCount() int {
	f.mutex.Lock()
	n := len(f.lines)
	f.mutex.Unlock()
	return n
}

// AddLine adds the line offset for a new line.
// The line offset must be larger than the offset for the previous line
// and smaller than the file size; otherwise the line offset is ignored.
func (f *File) AddLine(offset int) {
	f.mutex.Lock()
	if i := len(f.lines); (i == 0 || f.lines[i-1] < offset) && offset < f.size {
		f.lines = append(f.lines, offset)
	}
	f.mutex.Unlock()
}

// MergeLine merges a line with the following line. It is akin to replacing
// the newline character at the end of the line with a space (to not change the
// remaining offsets). To obtain the line number, consult e.g. [Position.Line].
// MergeLine will panic if given an invalid line number.
func (f *File) MergeLine(line int) {
	if line < 1 {
		panic(fmt.Sprintf("invalid line number %d (should be >= 1)", line))
	}
	f.mutex.Lock()
	defer f.mutex.Unlock()
	if line >= len(f.lines) {
		panic(fmt.Sprintf("invalid line number %d (should be < %d)", line, len(f.lines)))
	}
	// To merge the line numbered <line> with the line numbered <line+1>,
	// we need to remove the entry in lines corresponding to the line
	// numbered <line+1>. The entry in lines corresponding to the line
	// numbered <line+1> is located at index <line>, since indices in lines
	// are 0-based and line numbers are 1-based.
	copy(f.lines[line:], f.lines[line+1:])
	f.lines = f.lines[:len(f.lines)-1]
}

// Lines returns the effective line offset table of the form described by [File.SetLines].
// Callers must not mutate the result.
func (f *File) Lines() []int {
	f.mutex.Lock()
	lines := f.lines
	f.mutex.Unlock()
	return lines
}

// SetLines sets the line offsets for a file and reports whether it succeeded.
// The line offsets are the offsets of the first character of each line;
// for instance for the content "ab\nc\n" the line offsets are {0, 3}.
// An empty file has an empty line offset table.
// Each line offset must be larger than the offset for the previous line
// and smaller than the file size; otherwise SetLines fails and returns
// false.
// Callers must not mutate the provided slice after SetLines returns.
func (f *File) SetLines(lines []int) bool {
	// verify validity of lines table
	size := f.size
	for i, offset := range lines {
		if i > 0 && offset <= lines[i-1] || size <= offset {
			return false
		}
	}

	// set lines table
	f.mutex.Lock()
	f.lines = lines
	f.mutex.Unlock()
	return true
}

// SetLinesForContent sets the line offsets for the given file content.
// It ignores position-altering //line comments.
func (f *File) SetLinesForContent(content []byte) {
	var lines []int
	line := 0
	for offset, b := range content {
		if line >= 0 {
			lines = append(lines, line)
		}
		line = -1
		if b == '\n' {
			line = offset + 1
		}
	}

	// set lines table
	f.mutex.Lock()
	f.lines = lines
	f.mutex.Unlock()
}

// LineStart returns the [Pos] value of the start of the specified line.
// It ignores any alternative positions set using [File.AddLineColumnInfo].
// LineStart panics if the 1-based line number is invalid.
func (f *File) LineStart(line int) Pos {
	if line < 1 {
		panic(fmt.Sprintf("invalid line number %d (should be >= 1)", line))
	}
	f.mutex.Lock()
	defer f.mutex.Unlock()
	if line > len(f.lines) {
		panic(fmt.Sprintf("invalid line number %d (should be < %d)", line, len(f.lines)))
	}
	return Pos(f.base + f.lines[line-1])
}

// A lineInfo object describes alternative file, line, and column
// number information (such as provided via a //line directive)
// for a given file offset.
type lineInfo struct {
	// fields are exported to make them accessible to gob
	Offset       int
	Filename     string
	Line, Column int
}

// AddLineInfo is like [File.AddLineColumnInfo] with a column = 1 argument.
// It is here for backward-compatibility for code prior to Go 1.11.
func (f *File) AddLineInfo(offset int, filename string, line int) {
	f.AddLineColumnInfo(offset, filename, line, 1)
}

// AddLineColumnInfo adds alternative file, line, and column number
// information for a given file offset. The offset must be larger
// than the offset for the previously added alternative line info
// and smaller than the file size; otherwise the information is
// ignored.
//
// AddLineColumnInfo is typically used to register alternative position
// information for line directives such as //line filename:line:column.
func (f *File) AddLineColumnInfo(offset int, filename string, line, column int) {
	f.mutex.Lock()
	if i := len(f.infos); (i == 0 || f.infos[i-1].Offset < offset) && offset < f.size {
		f.infos = append(f.infos, lineInfo{offset, filename, line, column})
	}
	f.mutex.Unlock()
}

// fixOffset fixes an out-of-bounds offset such that 0 <= offset <= f.size.
func (f *File) fixOffset(offset int) int {
	switch {
	case offset < 0:
		if !debug {
			return 0
		}
	case offset > f.size:
		if !debug {
			return f.size
		}
	default:
		return offset
	}

	// only generate this code if needed
	if debug {
		panic(fmt.Sprintf("offset %d out of bounds [%d, %d] (position %d out of bounds [%d, %d])",
			0 /* for symmetry */, offset, f.size,
			f.base+offset, f.base, f.base+f.size))
	}
	return 0
}

// Pos returns the Pos value for the given file offset.
//
// If offset is negative, the result is the file's start
// position; if the offset is too large, the result is
// the file's end position (see also go.dev/issue/57490).
//
// The following invariant, though not true for Pos values
// in general, holds for the result p:
// f.Pos(f.Offset(p)) == p.
func (f *File) Pos(offset int) Pos {
	return Pos(f.base + f.fixOffset(offset))
}

// Offset returns the offset for the given file position p.
//
// If p is before the file's start position (or if p is NoPos),
// the result is 0; if p is past the file's end position,
// the result is the file size (see also go.dev/issue/57490).
//
// The following invariant, though not true for offset values
// in general, holds for the result offset:
// f.Offset(f.Pos(offset)) == offset
func (f *File) Offset(p Pos) int {
	return f.fixOffset(int(p) - f.base)
}

// Line returns the line number for the given file position p;
// p must be a [Pos] value in that file or [NoPos].
func (f *File) Line(p Pos) int {
	return f.Position(p).Line
}

func searchLineInfos(a []lineInfo, x int) int {
	i, found := slices.BinarySearchFunc(a, x, func(a lineInfo, x int) int {
		return cmp.Compare(a.Offset, x)
	})
	if !found {
		// We want the lineInfo containing x, but if we didn't
		// find x then i is the next one.
		i--
	}
	return i
}

// unpack returns the filename and line and column number for a file offset.
// If adjusted is set, unpack will return the filename and line information
// possibly adjusted by //line comments; otherwise those comments are ignored.
func (f *File) unpack(offset int, adjusted bool) (filename string, line, column int) {
	f.mutex.Lock()
	filename = f.name
	if i := searchInts(f.lines, offset); i >= 0 {
		line, column = i+1, offset-f.lines[i]+1
	}
	if adjusted && len(f.infos) > 0 {
		// few files have extra line infos
		if i := searchLineInfos(f.infos, offset); i >= 0 {
			alt := &f.infos[i]
			filename = alt.Filename
			if i := searchInts(f.lines, alt.Offset); i >= 0 {
				// i+1 is the line at which the alternative position was recorded
				d := line - (i + 1) // line distance from alternative position base
				line = alt.Line + d
				if alt.Column == 0 {
					// alternative column is unknown => relative column is unknown
					// (the current specification for line directives requires
					// this to apply until the next PosBase/line directive,
					// not just until the new newline)
					column = 0
				} else if d == 0 {
					// the alternative position base is on the current line
					// => column is relative to alternative column
					column = alt.Column + (offset - alt.Offset)
				}
			}
		}
	}
	// TODO(mvdan): move Unlock back under Lock with a defer statement once
	// https://go.dev/issue/38471 is fixed to remove the performance penalty.
	f.mutex.Unlock()
	return
}

func (f *File) position(p Pos, adjusted bool) (pos Position) {
	offset := f.fixOffset(int(p) - f.base)
	pos.Offset = offset
	pos.Filename, pos.Line, pos.Column = f.unpack(offset, adjusted)
	return
}

// PositionFor returns the Position value for the given file position p.
// If p is out of bounds, it is adjusted to match the File.Offset behavior.
// If adjusted is set, the position may be adjusted by position-altering
// //line comments; otherwise those comments are ignored.
// p must be a Pos value in f or NoPos.
func (f *File) PositionFor(p Pos, adjusted bool) (pos Position) {
	if p != NoPos {
		pos = f.position(p, adjusted)
	}
	return
}

// Position returns the Position value for the given file position p.
// If p is out of bounds, it is adjusted to match the File.Offset behavior.
// Calling f.Position(p) is equivalent to calling f.PositionFor(p, true).
func (f *File) Position(p Pos) (pos Position) {
	return f.PositionFor(p, true)
}

// -----------------------------------------------------------------------------
// FileSet

// A FileSet represents a set of source files.
// Methods of file sets are synchronized; multiple goroutines
// may invoke them concurrently.
//
// The byte offsets for each file in a file set are mapped into
// distinct (integer) intervals, one interval [base, base+size]
// per file. [FileSet.Base] represents the first byte in the file, and size
// is the corresponding file size. A [Pos] value is a value in such
// an interval. By determining the interval a [Pos] value belongs
// to, the file, its file base, and thus the byte offset (position)
// the [Pos] value is representing can be computed.
//
// When adding a new file, a file base must be provided. That can
// be any integer value that is past the end of any interval of any
// file already in the file set. For convenience, [FileSet.Base] provides
// such a value, which is simply the end of the Pos interval of the most
// recently added file, plus one. Unless there is a need to extend an
// interval later, using the [FileSet.Base] should be used as argument
// for [FileSet.AddFile].
//
// A [File] may be removed from a FileSet when it is no longer needed.
// This may reduce memory usage in a long-running application.
type FileSet struct {
	mutex sync.RWMutex         // protects the file set
	base  int                  // base offset for the next file
	tree  tree                 // tree of files in ascending base order
	last  atomic.Pointer[File] // cache of last file looked up
}

// NewFileSet creates a new file set.
func NewFileSet() *FileSet {
	return &FileSet{
		base: 1, // 0 == NoPos
	}
}

// Base returns the minimum base offset that must be provided to
// [FileSet.AddFile] when adding the next file.
func (s *FileSet) Base() int {
	s.mutex.RLock()
	b := s.base
	s.mutex.RUnlock()
	return b
}

// AddFile adds a new file with a given filename, base offset, and file size
// to the file set s and returns the file. Multiple files may have the same
// name. The base offset must not be smaller than the [FileSet.Base], and
// size must not be negative. As a special case, if a negative base is provided,
// the current value of the [FileSet.Base] is used instead.
//
// Adding the file will set the file set's [FileSet.Base] value to base + size + 1
// as the minimum base value for the next file. The following relationship
// exists between a [Pos] value p for a given file offset offs:
//
//	int(p) = base + offs
//
// with offs in the range [0, size] and thus p in the range [base, base+size].
// For convenience, [File.Pos] may be used to create file-specific position
// values from a file offset.
func (s *FileSet) AddFile(filename string, base, size int) *File {
	// Allocate f outside the critical section.
	f := &File{name: filename, size: size, lines: []int{0}}

	s.mutex.Lock()
	defer s.mutex.Unlock()
	if base < 0 {
		base = s.base
	}
	if base < s.base {
		panic(fmt.Sprintf("invalid base %d (should be >= %d)", base, s.base))
	}
	f.base = base
	if size < 0 {
		panic(fmt.Sprintf("invalid size %d (should be >= 0)", size))
	}
	// base >= s.base && size >= 0
	base += size + 1 // +1 because EOF also has a position
	if base < 0 {
		panic("token.Pos offset overflow (> 2G of source code in file set)")
	}
	// add the file to the file set
	s.base = base
	s.tree.add(f)
	s.last.Store(f)
	return f
}

// AddExistingFiles adds the specified files to the
// FileSet if they are not already present.
// The caller must ensure that no pair of Files that
// would appear in the resulting FileSet overlap.
func (s *FileSet) AddExistingFiles(files ...*File) {
	// This function cannot be implemented as:
	//
	//	for _, file := range files {
	//		if prev := fset.File(token.Pos(file.Base())); prev != nil {
	//			if prev != file {
	//				panic("FileSet contains a different file at the same base")
	//			}
	//			continue
	//		}
	//		file2 := fset.AddFile(file.Name(), file.Base(), file.Size())
	//		file2.SetLines(file.Lines())
	//	}
	//
	// because all calls to AddFile must be in increasing order.
	// AddExistingFiles lets us augment an existing FileSet
	// sequentially, so long as all sets of files have disjoint ranges.
	// This approach also does not preserve line directives.

	s.mutex.Lock()
	defer s.mutex.Unlock()

	for _, f := range files {
		s.tree.add(f)
		s.base = max(s.base, f.Base()+f.Size()+1)
	}
}

// RemoveFile removes a file from the [FileSet] so that subsequent
// queries for its [Pos] interval yield a negative result.
// This reduces the memory usage of a long-lived [FileSet] that
// encounters an unbounded stream of files.
//
// Removing a file that does not belong to the set has no effect.
func (s *FileSet) RemoveFile(file *File) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	s.last.CompareAndSwap(file, nil) // clear last file cache

	pn, _ := s.tree.locate(file.key())
	if *pn != nil && (*pn).file == file {
		s.tree.delete(pn)
	}
}

// Iterate calls yield for the files in the file set in ascending Base
// order until yield returns false.
func (s *FileSet) Iterate(yield func(*File) bool) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	// Unlock around user code.
	// The iterator is robust to modification by yield.
	// Avoid range here, so we can use defer.
	s.tree.all()(func(f *File) bool {
		s.mutex.RUnlock()
		defer s.mutex.RLock()
		return yield(f)
	})
}

func (s *FileSet) file(p Pos) *File {
	// common case: p is in last file.
	if f := s.last.Load(); f != nil && f.base <= int(p) && int(p) <= f.base+f.size {
		return f
	}

	s.mutex.RLock()
	defer s.mutex.RUnlock()

	pn, _ := s.tree.locate(key{int(p), int(p)})
	if n := *pn; n != nil {
		// Update cache of last file. A race is ok,
		// but an exclusive lock causes heavy contention.
		s.last.Store(n.file)
		return n.file
	}
	return nil
}

// File returns the file that contains the position p.
// If no such file is found (for instance for p == [NoPos]),
// the result is nil.
func (s *FileSet) File(p Pos) (f *File) {
	if p != NoPos {
		f = s.file(p)
	}
	return
}

// PositionFor converts a [Pos] p in the fileset into a [Position] value.
// If adjusted is set, the position may be adjusted by position-altering
// //line comments; otherwise those comments are ignored.
// p must be a [Pos] value in s or [NoPos].
func (s *FileSet) PositionFor(p Pos, adjusted bool) (pos Position) {
	if p != NoPos {
		if f := s.file(p); f != nil {
			return f.position(p, adjusted)
		}
	}
	return
}

// Position converts a [Pos] p in the fileset into a Position value.
// Calling s.Position(p) is equivalent to calling s.PositionFor(p, true).
func (s *FileSet) Position(p Pos) (pos Position) {
	return s.PositionFor(p, true)
}

// -----------------------------------------------------------------------------
// Helper functions

func searchInts(a []int, x int) int {
	// This function body is a manually inlined version of:
	//
	//   return sort.Search(len(a), func(i int) bool { return a[i] > x }) - 1
	//
	// With better compiler optimizations, this may not be needed in the
	// future, but at the moment this change improves the go/printer
	// benchmark performance by ~30%. This has a direct impact on the
	// speed of gofmt and thus seems worthwhile (2011-04-29).
	// TODO(gri): Remove this when compilers have caught up.
	i, j := 0, len(a)
	for i < j {
		h := int(uint(i+j) >> 1) // avoid overflow when computing h
		// i ≤ h < j
		if a[h] <= x {
			i = h + 1
		} else {
			j = h
		}
	}
	return i - 1
}
