// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains the infrastructure to create an
// identifier and full-text index for a set of Go files.
//
// Algorithm for identifier index:
// - traverse all .go files of the file tree specified by root
// - for each identifier (word) encountered, collect all occurrences (spots)
//   into a list; this produces a list of spots for each word
// - reduce the lists: from a list of spots to a list of FileRuns,
//   and from a list of FileRuns into a list of PakRuns
// - make a HitList from the PakRuns
//
// Details:
// - keep two lists per word: one containing package-level declarations
//   that have snippets, and one containing all other spots
// - keep the snippets in a separate table indexed by snippet index
//   and store the snippet index in place of the line number in a SpotInfo
//   (the line number for spots with snippets is stored in the snippet)
// - at the end, create lists of alternative spellings for a given
//   word
//
// Algorithm for full text index:
// - concatenate all source code in a byte buffer (in memory)
// - add the files to a file set in lockstep as they are added to the byte
//   buffer such that a byte buffer offset corresponds to the Pos value for
//   that file location
// - create a suffix array from the concatenated sources
//
// String lookup in full text index:
// - use the suffix array to lookup a string's offsets - the offsets
//   correspond to the Pos values relative to the file set
// - translate the Pos values back into file and line information and
//   sort the result

package godoc

import (
	"bufio"
	"bytes"
	"encoding/gob"
	"errors"
	"fmt"
	"go/ast"
	"go/doc"
	"go/parser"
	"go/token"
	"index/suffixarray"
	"io"
	"log"
	"math"
	"os"
	pathpkg "path"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode"

	"golang.org/x/tools/godoc/util"
	"golang.org/x/tools/godoc/vfs"
)

// ----------------------------------------------------------------------------
// InterfaceSlice is a helper type for sorting interface
// slices according to some slice-specific sort criteria.

type comparer func(x, y interface{}) bool

type interfaceSlice struct {
	slice []interface{}
	less  comparer
}

// ----------------------------------------------------------------------------
// RunList

// A RunList is a list of entries that can be sorted according to some
// criteria. A RunList may be compressed by grouping "runs" of entries
// which are equal (according to the sort criteria) into a new RunList of
// runs. For instance, a RunList containing pairs (x, y) may be compressed
// into a RunList containing pair runs (x, {y}) where each run consists of
// a list of y's with the same x.
type RunList []interface{}

func (h RunList) sort(less comparer) {
	sort.Sort(&interfaceSlice{h, less})
}

func (p *interfaceSlice) Len() int           { return len(p.slice) }
func (p *interfaceSlice) Less(i, j int) bool { return p.less(p.slice[i], p.slice[j]) }
func (p *interfaceSlice) Swap(i, j int)      { p.slice[i], p.slice[j] = p.slice[j], p.slice[i] }

// Compress entries which are the same according to a sort criteria
// (specified by less) into "runs".
func (h RunList) reduce(less comparer, newRun func(h RunList) interface{}) RunList {
	if len(h) == 0 {
		return nil
	}
	// len(h) > 0

	// create runs of entries with equal values
	h.sort(less)

	// for each run, make a new run object and collect them in a new RunList
	var hh RunList
	i, x := 0, h[0]
	for j, y := range h {
		if less(x, y) {
			hh = append(hh, newRun(h[i:j]))
			i, x = j, h[j] // start a new run
		}
	}
	// add final run, if any
	if i < len(h) {
		hh = append(hh, newRun(h[i:]))
	}

	return hh
}

// ----------------------------------------------------------------------------
// KindRun

// Debugging support. Disable to see multiple entries per line.
const removeDuplicates = true

// A KindRun is a run of SpotInfos of the same kind in a given file.
// The kind (3 bits) is stored in each SpotInfo element; to find the
// kind of a KindRun, look at any of its elements.
type KindRun []SpotInfo

// KindRuns are sorted by line number or index. Since the isIndex bit
// is always the same for all infos in one list we can compare lori's.
func (k KindRun) Len() int           { return len(k) }
func (k KindRun) Less(i, j int) bool { return k[i].Lori() < k[j].Lori() }
func (k KindRun) Swap(i, j int)      { k[i], k[j] = k[j], k[i] }

// FileRun contents are sorted by Kind for the reduction into KindRuns.
func lessKind(x, y interface{}) bool { return x.(SpotInfo).Kind() < y.(SpotInfo).Kind() }

// newKindRun allocates a new KindRun from the SpotInfo run h.
func newKindRun(h RunList) interface{} {
	run := make(KindRun, len(h))
	for i, x := range h {
		run[i] = x.(SpotInfo)
	}

	// Spots were sorted by file and kind to create this run.
	// Within this run, sort them by line number or index.
	sort.Sort(run)

	if removeDuplicates {
		// Since both the lori and kind field must be
		// same for duplicates, and since the isIndex
		// bit is always the same for all infos in one
		// list we can simply compare the entire info.
		k := 0
		prev := SpotInfo(math.MaxUint32) // an unlikely value
		for _, x := range run {
			if x != prev {
				run[k] = x
				k++
				prev = x
			}
		}
		run = run[0:k]
	}

	return run
}

// ----------------------------------------------------------------------------
// FileRun

// A Pak describes a Go package.
type Pak struct {
	Path string // path of directory containing the package
	Name string // package name as declared by package clause
}

// Paks are sorted by name (primary key) and by import path (secondary key).
func (p *Pak) less(q *Pak) bool {
	return p.Name < q.Name || p.Name == q.Name && p.Path < q.Path
}

// A File describes a Go file.
type File struct {
	Name string // directory-local file name
	Pak  *Pak   // the package to which the file belongs
}

// Path returns the file path of f.
func (f *File) Path() string {
	return pathpkg.Join(f.Pak.Path, f.Name)
}

// A Spot describes a single occurrence of a word.
type Spot struct {
	File *File
	Info SpotInfo
}

// A FileRun is a list of KindRuns belonging to the same file.
type FileRun struct {
	File   *File
	Groups []KindRun
}

// Spots are sorted by file path for the reduction into FileRuns.
func lessSpot(x, y interface{}) bool {
	fx := x.(Spot).File
	fy := y.(Spot).File
	// same as "return fx.Path() < fy.Path()" but w/o computing the file path first
	px := fx.Pak.Path
	py := fy.Pak.Path
	return px < py || px == py && fx.Name < fy.Name
}

// newFileRun allocates a new FileRun from the Spot run h.
func newFileRun(h RunList) interface{} {
	file := h[0].(Spot).File

	// reduce the list of Spots into a list of KindRuns
	h1 := make(RunList, len(h))
	for i, x := range h {
		h1[i] = x.(Spot).Info
	}
	h2 := h1.reduce(lessKind, newKindRun)

	// create the FileRun
	groups := make([]KindRun, len(h2))
	for i, x := range h2 {
		groups[i] = x.(KindRun)
	}
	return &FileRun{file, groups}
}

// ----------------------------------------------------------------------------
// PakRun

// A PakRun describes a run of *FileRuns of a package.
type PakRun struct {
	Pak   *Pak
	Files []*FileRun
}

// Sorting support for files within a PakRun.
func (p *PakRun) Len() int           { return len(p.Files) }
func (p *PakRun) Less(i, j int) bool { return p.Files[i].File.Name < p.Files[j].File.Name }
func (p *PakRun) Swap(i, j int)      { p.Files[i], p.Files[j] = p.Files[j], p.Files[i] }

// FileRuns are sorted by package for the reduction into PakRuns.
func lessFileRun(x, y interface{}) bool {
	return x.(*FileRun).File.Pak.less(y.(*FileRun).File.Pak)
}

// newPakRun allocates a new PakRun from the *FileRun run h.
func newPakRun(h RunList) interface{} {
	pak := h[0].(*FileRun).File.Pak
	files := make([]*FileRun, len(h))
	for i, x := range h {
		files[i] = x.(*FileRun)
	}
	run := &PakRun{pak, files}
	sort.Sort(run) // files were sorted by package; sort them by file now
	return run
}

// ----------------------------------------------------------------------------
// HitList

// A HitList describes a list of PakRuns.
type HitList []*PakRun

// PakRuns are sorted by package.
func lessPakRun(x, y interface{}) bool { return x.(*PakRun).Pak.less(y.(*PakRun).Pak) }

func reduce(h0 RunList) HitList {
	// reduce a list of Spots into a list of FileRuns
	h1 := h0.reduce(lessSpot, newFileRun)
	// reduce a list of FileRuns into a list of PakRuns
	h2 := h1.reduce(lessFileRun, newPakRun)
	// sort the list of PakRuns by package
	h2.sort(lessPakRun)
	// create a HitList
	h := make(HitList, len(h2))
	for i, p := range h2 {
		h[i] = p.(*PakRun)
	}
	return h
}

// filter returns a new HitList created by filtering
// all PakRuns from h that have a matching pakname.
func (h HitList) filter(pakname string) HitList {
	var hh HitList
	for _, p := range h {
		if p.Pak.Name == pakname {
			hh = append(hh, p)
		}
	}
	return hh
}

// ----------------------------------------------------------------------------
// AltWords

type wordPair struct {
	canon string // canonical word spelling (all lowercase)
	alt   string // alternative spelling
}

// An AltWords describes a list of alternative spellings for a
// canonical (all lowercase) spelling of a word.
type AltWords struct {
	Canon string   // canonical word spelling (all lowercase)
	Alts  []string // alternative spelling for the same word
}

// wordPairs are sorted by their canonical spelling.
func lessWordPair(x, y interface{}) bool { return x.(*wordPair).canon < y.(*wordPair).canon }

// newAltWords allocates a new AltWords from the *wordPair run h.
func newAltWords(h RunList) interface{} {
	canon := h[0].(*wordPair).canon
	alts := make([]string, len(h))
	for i, x := range h {
		alts[i] = x.(*wordPair).alt
	}
	return &AltWords{canon, alts}
}

func (a *AltWords) filter(s string) *AltWords {
	var alts []string
	for _, w := range a.Alts {
		if w != s {
			alts = append(alts, w)
		}
	}
	if len(alts) > 0 {
		return &AltWords{a.Canon, alts}
	}
	return nil
}

// Ident stores information about external identifiers in order to create
// links to package documentation.
type Ident struct {
	Path    string // e.g. "net/http"
	Package string // e.g. "http"
	Name    string // e.g. "NewRequest"
	Doc     string // e.g. "NewRequest returns a new Request..."
}

// byImportCount sorts the given slice of Idents by the import
// counts of the packages to which they belong.
type byImportCount struct {
	Idents      []Ident
	ImportCount map[string]int
}

func (ic byImportCount) Len() int {
	return len(ic.Idents)
}

func (ic byImportCount) Less(i, j int) bool {
	ri := ic.ImportCount[ic.Idents[i].Path]
	rj := ic.ImportCount[ic.Idents[j].Path]
	if ri == rj {
		return ic.Idents[i].Path < ic.Idents[j].Path
	}
	return ri > rj
}

func (ic byImportCount) Swap(i, j int) {
	ic.Idents[i], ic.Idents[j] = ic.Idents[j], ic.Idents[i]
}

func (ic byImportCount) String() string {
	buf := bytes.NewBuffer([]byte("["))
	for _, v := range ic.Idents {
		buf.WriteString(fmt.Sprintf("\n\t%s, %s (%d)", v.Path, v.Name, ic.ImportCount[v.Path]))
	}
	buf.WriteString("\n]")
	return buf.String()
}

// filter creates a new Ident list where the results match the given
// package name.
func (ic byImportCount) filter(pakname string) []Ident {
	if ic.Idents == nil {
		return nil
	}
	var res []Ident
	for _, i := range ic.Idents {
		if i.Package == pakname {
			res = append(res, i)
		}
	}
	return res
}

// top returns the top n identifiers.
func (ic byImportCount) top(n int) []Ident {
	if len(ic.Idents) > n {
		return ic.Idents[:n]
	}
	return ic.Idents
}

// ----------------------------------------------------------------------------
// Indexer

type IndexResult struct {
	Decls  RunList // package-level declarations (with snippets)
	Others RunList // all other occurrences
}

// Statistics provides statistics information for an index.
type Statistics struct {
	Bytes int // total size of indexed source files
	Files int // number of indexed source files
	Lines int // number of lines (all files)
	Words int // number of different identifiers
	Spots int // number of identifier occurrences
}

// An Indexer maintains the data structures and provides the machinery
// for indexing .go files under a file tree. It implements the path.Visitor
// interface for walking file trees, and the ast.Visitor interface for
// walking Go ASTs.
type Indexer struct {
	c          *Corpus
	fset       *token.FileSet // file set for all indexed files
	fsOpenGate chan bool      // send pre fs.Open; receive on close

	mu            sync.Mutex              // guards all the following
	sources       bytes.Buffer            // concatenated sources
	strings       map[string]string       // interned string
	packages      map[Pak]*Pak            // interned *Paks
	words         map[string]*IndexResult // RunLists of Spots
	snippets      []*Snippet              // indices are stored in SpotInfos
	current       *token.File             // last file added to file set
	file          *File                   // AST for current file
	decl          ast.Decl                // AST for current decl
	stats         Statistics
	throttle      *util.Throttle
	importCount   map[string]int                 // package path ("net/http") => count
	packagePath   map[string]map[string]bool     // "template" => "text/template" => true
	exports       map[string]map[string]SpotKind // "net/http" => "ListenAndServe" => FuncDecl
	curPkgExports map[string]SpotKind
	idents        map[SpotKind]map[string][]Ident // kind => name => list of Idents
}

func (x *Indexer) intern(s string) string {
	if s, ok := x.strings[s]; ok {
		return s
	}
	x.strings[s] = s
	return s
}

func (x *Indexer) lookupPackage(path, name string) *Pak {
	// In the source directory tree, more than one package may
	// live in the same directory. For the packages map, construct
	// a key that includes both the directory path and the package
	// name.
	key := Pak{Path: x.intern(path), Name: x.intern(name)}
	pak := x.packages[key]
	if pak == nil {
		pak = &key
		x.packages[key] = pak
	}
	return pak
}

func (x *Indexer) addSnippet(s *Snippet) int {
	index := len(x.snippets)
	x.snippets = append(x.snippets, s)
	return index
}

func (x *Indexer) visitIdent(kind SpotKind, id *ast.Ident) {
	if id == nil {
		return
	}
	name := x.intern(id.Name)

	switch kind {
	case TypeDecl, FuncDecl, ConstDecl, VarDecl:
		x.curPkgExports[name] = kind
	}

	lists, found := x.words[name]
	if !found {
		lists = new(IndexResult)
		x.words[name] = lists
	}

	if kind == Use || x.decl == nil {
		if x.c.IndexGoCode {
			// not a declaration or no snippet required
			info := makeSpotInfo(kind, x.current.Line(id.Pos()), false)
			lists.Others = append(lists.Others, Spot{x.file, info})
		}
	} else {
		// a declaration with snippet
		index := x.addSnippet(NewSnippet(x.fset, x.decl, id))
		info := makeSpotInfo(kind, index, true)
		lists.Decls = append(lists.Decls, Spot{x.file, info})
	}

	x.stats.Spots++
}

func (x *Indexer) visitFieldList(kind SpotKind, flist *ast.FieldList) {
	for _, f := range flist.List {
		x.decl = nil // no snippets for fields
		for _, name := range f.Names {
			x.visitIdent(kind, name)
		}
		ast.Walk(x, f.Type)
		// ignore tag - not indexed at the moment
	}
}

func (x *Indexer) visitSpec(kind SpotKind, spec ast.Spec) {
	switch n := spec.(type) {
	case *ast.ImportSpec:
		x.visitIdent(ImportDecl, n.Name)
		if n.Path != nil {
			if imp, err := strconv.Unquote(n.Path.Value); err == nil {
				x.importCount[x.intern(imp)]++
			}
		}

	case *ast.ValueSpec:
		for _, n := range n.Names {
			x.visitIdent(kind, n)
		}
		ast.Walk(x, n.Type)
		for _, v := range n.Values {
			ast.Walk(x, v)
		}

	case *ast.TypeSpec:
		x.visitIdent(TypeDecl, n.Name)
		ast.Walk(x, n.Type)
	}
}

func (x *Indexer) visitGenDecl(decl *ast.GenDecl) {
	kind := VarDecl
	if decl.Tok == token.CONST {
		kind = ConstDecl
	}
	x.decl = decl
	for _, s := range decl.Specs {
		x.visitSpec(kind, s)
	}
}

func (x *Indexer) Visit(node ast.Node) ast.Visitor {
	switch n := node.(type) {
	case nil:
		// nothing to do

	case *ast.Ident:
		x.visitIdent(Use, n)

	case *ast.FieldList:
		x.visitFieldList(VarDecl, n)

	case *ast.InterfaceType:
		x.visitFieldList(MethodDecl, n.Methods)

	case *ast.DeclStmt:
		// local declarations should only be *ast.GenDecls;
		// ignore incorrect ASTs
		if decl, ok := n.Decl.(*ast.GenDecl); ok {
			x.decl = nil // no snippets for local declarations
			x.visitGenDecl(decl)
		}

	case *ast.GenDecl:
		x.decl = n
		x.visitGenDecl(n)

	case *ast.FuncDecl:
		kind := FuncDecl
		if n.Recv != nil {
			kind = MethodDecl
			ast.Walk(x, n.Recv)
		}
		x.decl = n
		x.visitIdent(kind, n.Name)
		ast.Walk(x, n.Type)
		if n.Body != nil {
			ast.Walk(x, n.Body)
		}

	case *ast.File:
		x.decl = nil
		x.visitIdent(PackageClause, n.Name)
		for _, d := range n.Decls {
			ast.Walk(x, d)
		}

	default:
		return x
	}

	return nil
}

// addFile adds a file to the index if possible and returns the file set file
// and the file's AST if it was successfully parsed as a Go file. If addFile
// failed (that is, if the file was not added), it returns file == nil.
func (x *Indexer) addFile(f vfs.ReadSeekCloser, filename string, goFile bool) (file *token.File, ast *ast.File) {
	defer f.Close()

	// The file set's base offset and x.sources size must be in lock-step;
	// this permits the direct mapping of suffix array lookup results to
	// corresponding Pos values.
	//
	// When a file is added to the file set, its offset base increases by
	// the size of the file + 1; and the initial base offset is 1. Add an
	// extra byte to the sources here.
	x.sources.WriteByte(0)

	// If the sources length doesn't match the file set base at this point
	// the file set implementation changed or we have another error.
	base := x.fset.Base()
	if x.sources.Len() != base {
		panic("internal error: file base incorrect")
	}

	// append file contents (src) to x.sources
	if _, err := x.sources.ReadFrom(f); err == nil {
		src := x.sources.Bytes()[base:]

		if goFile {
			// parse the file and in the process add it to the file set
			if ast, err = parser.ParseFile(x.fset, filename, src, parser.ParseComments); err == nil {
				file = x.fset.File(ast.Pos()) // ast.Pos() is inside the file
				return
			}
			// file has parse errors, and the AST may be incorrect -
			// set lines information explicitly and index as ordinary
			// text file (cannot fall through to the text case below
			// because the file has already been added to the file set
			// by the parser)
			file = x.fset.File(token.Pos(base)) // token.Pos(base) is inside the file
			file.SetLinesForContent(src)
			ast = nil
			return
		}

		if util.IsText(src) {
			// only add the file to the file set (for the full text index)
			file = x.fset.AddFile(filename, x.fset.Base(), len(src))
			file.SetLinesForContent(src)
			return
		}
	}

	// discard possibly added data
	x.sources.Truncate(base - 1) // -1 to remove added byte 0 since no file was added
	return
}

// Design note: Using an explicit white list of permitted files for indexing
// makes sure that the important files are included and massively reduces the
// number of files to index. The advantage over a blacklist is that unexpected
// (non-blacklisted) files won't suddenly explode the index.

// Files are whitelisted if they have a file name or extension
// present as key in whitelisted.
var whitelisted = map[string]bool{
	".bash":        true,
	".c":           true,
	".cc":          true,
	".cpp":         true,
	".cxx":         true,
	".css":         true,
	".go":          true,
	".goc":         true,
	".h":           true,
	".hh":          true,
	".hpp":         true,
	".hxx":         true,
	".html":        true,
	".js":          true,
	".out":         true,
	".py":          true,
	".s":           true,
	".sh":          true,
	".txt":         true,
	".xml":         true,
	"AUTHORS":      true,
	"CONTRIBUTORS": true,
	"LICENSE":      true,
	"Makefile":     true,
	"PATENTS":      true,
	"README":       true,
}

// isWhitelisted returns true if a file is on the list
// of "permitted" files for indexing. The filename must
// be the directory-local name of the file.
func isWhitelisted(filename string) bool {
	key := pathpkg.Ext(filename)
	if key == "" {
		// file has no extension - use entire filename
		key = filename
	}
	return whitelisted[key]
}

func (x *Indexer) indexDocs(dirname string, filename string, astFile *ast.File) {
	pkgName := x.intern(astFile.Name.Name)
	if pkgName == "main" {
		return
	}
	pkgPath := x.intern(strings.TrimPrefix(strings.TrimPrefix(dirname, "/src/"), "pkg/"))
	astPkg := ast.Package{
		Name: pkgName,
		Files: map[string]*ast.File{
			filename: astFile,
		},
	}
	var m doc.Mode
	docPkg := doc.New(&astPkg, dirname, m)
	addIdent := func(sk SpotKind, name string, docstr string) {
		if x.idents[sk] == nil {
			x.idents[sk] = make(map[string][]Ident)
		}
		name = x.intern(name)
		x.idents[sk][name] = append(x.idents[sk][name], Ident{
			Path:    pkgPath,
			Package: pkgName,
			Name:    name,
			Doc:     doc.Synopsis(docstr),
		})
	}

	if x.idents[PackageClause] == nil {
		x.idents[PackageClause] = make(map[string][]Ident)
	}
	// List of words under which the package identifier will be stored.
	// This includes the package name and the components of the directory
	// in which it resides.
	words := strings.Split(pathpkg.Dir(pkgPath), "/")
	if words[0] == "." {
		words = []string{}
	}
	name := x.intern(docPkg.Name)
	synopsis := doc.Synopsis(docPkg.Doc)
	words = append(words, name)
	pkgIdent := Ident{
		Path:    pkgPath,
		Package: pkgName,
		Name:    name,
		Doc:     synopsis,
	}
	for _, word := range words {
		word = x.intern(word)
		found := false
		pkgs := x.idents[PackageClause][word]
		for i, p := range pkgs {
			if p.Path == pkgPath {
				if docPkg.Doc != "" {
					p.Doc = synopsis
					pkgs[i] = p
				}
				found = true
				break
			}
		}
		if !found {
			x.idents[PackageClause][word] = append(x.idents[PackageClause][word], pkgIdent)
		}
	}

	for _, c := range docPkg.Consts {
		for _, name := range c.Names {
			addIdent(ConstDecl, name, c.Doc)
		}
	}
	for _, t := range docPkg.Types {
		addIdent(TypeDecl, t.Name, t.Doc)
		for _, c := range t.Consts {
			for _, name := range c.Names {
				addIdent(ConstDecl, name, c.Doc)
			}
		}
		for _, v := range t.Vars {
			for _, name := range v.Names {
				addIdent(VarDecl, name, v.Doc)
			}
		}
		for _, f := range t.Funcs {
			addIdent(FuncDecl, f.Name, f.Doc)
		}
		for _, f := range t.Methods {
			addIdent(MethodDecl, f.Name, f.Doc)
			// Change the name of methods to be "<typename>.<methodname>".
			// They will still be indexed as <methodname>.
			idents := x.idents[MethodDecl][f.Name]
			idents[len(idents)-1].Name = x.intern(t.Name + "." + f.Name)
		}
	}
	for _, v := range docPkg.Vars {
		for _, name := range v.Names {
			addIdent(VarDecl, name, v.Doc)
		}
	}
	for _, f := range docPkg.Funcs {
		addIdent(FuncDecl, f.Name, f.Doc)
	}
}

func (x *Indexer) indexGoFile(dirname string, filename string, file *token.File, astFile *ast.File) {
	pkgName := astFile.Name.Name

	if x.c.IndexGoCode {
		x.current = file
		pak := x.lookupPackage(dirname, pkgName)
		x.file = &File{filename, pak}
		ast.Walk(x, astFile)
	}

	if x.c.IndexDocs {
		// Test files are already filtered out in visitFile if IndexGoCode and
		// IndexFullText are false.  Otherwise, check here.
		isTestFile := (x.c.IndexGoCode || x.c.IndexFullText) &&
			(strings.HasSuffix(filename, "_test.go") || strings.HasPrefix(dirname, "/test/"))
		if !isTestFile {
			x.indexDocs(dirname, filename, astFile)
		}
	}

	ppKey := x.intern(pkgName)
	if _, ok := x.packagePath[ppKey]; !ok {
		x.packagePath[ppKey] = make(map[string]bool)
	}
	pkgPath := x.intern(strings.TrimPrefix(strings.TrimPrefix(dirname, "/src/"), "pkg/"))
	x.packagePath[ppKey][pkgPath] = true

	// Merge in exported symbols found walking this file into
	// the map for that package.
	if len(x.curPkgExports) > 0 {
		dest, ok := x.exports[pkgPath]
		if !ok {
			dest = make(map[string]SpotKind)
			x.exports[pkgPath] = dest
		}
		for k, v := range x.curPkgExports {
			dest[k] = v
		}
	}
}

func (x *Indexer) visitFile(dirname string, fi os.FileInfo) {
	if fi.IsDir() || !x.c.IndexEnabled {
		return
	}

	filename := pathpkg.Join(dirname, fi.Name())
	goFile := isGoFile(fi)

	switch {
	case x.c.IndexFullText:
		if !isWhitelisted(fi.Name()) {
			return
		}
	case x.c.IndexGoCode:
		if !goFile {
			return
		}
	case x.c.IndexDocs:
		if !goFile ||
			strings.HasSuffix(fi.Name(), "_test.go") ||
			strings.HasPrefix(dirname, "/test/") {
			return
		}
	default:
		// No indexing turned on.
		return
	}

	x.fsOpenGate <- true
	defer func() { <-x.fsOpenGate }()

	// open file
	f, err := x.c.fs.Open(filename)
	if err != nil {
		return
	}

	x.mu.Lock()
	defer x.mu.Unlock()

	x.throttle.Throttle()

	x.curPkgExports = make(map[string]SpotKind)
	file, fast := x.addFile(f, filename, goFile)
	if file == nil {
		return // addFile failed
	}

	if fast != nil {
		x.indexGoFile(dirname, fi.Name(), file, fast)
	}

	// update statistics
	x.stats.Bytes += file.Size()
	x.stats.Files++
	x.stats.Lines += file.LineCount()
}

// indexOptions contains information that affects the contents of an index.
type indexOptions struct {
	// Docs provides documentation search results.
	// It is only consulted if IndexEnabled is true.
	// The default values is true.
	Docs bool

	// GoCode provides Go source code search results.
	// It is only consulted if IndexEnabled is true.
	// The default values is true.
	GoCode bool

	// FullText provides search results from all files.
	// It is only consulted if IndexEnabled is true.
	// The default values is true.
	FullText bool

	// MaxResults optionally specifies the maximum results for indexing.
	// The default is 1000.
	MaxResults int
}

// ----------------------------------------------------------------------------
// Index

type LookupResult struct {
	Decls  HitList // package-level declarations (with snippets)
	Others HitList // all other occurrences
}

type Index struct {
	fset        *token.FileSet           // file set used during indexing; nil if no textindex
	suffixes    *suffixarray.Index       // suffixes for concatenated sources; nil if no textindex
	words       map[string]*LookupResult // maps words to hit lists
	alts        map[string]*AltWords     // maps canonical(words) to lists of alternative spellings
	snippets    []*Snippet               // all snippets, indexed by snippet index
	stats       Statistics
	importCount map[string]int                 // package path ("net/http") => count
	packagePath map[string]map[string]bool     // "template" => "text/template" => true
	exports     map[string]map[string]SpotKind // "net/http" => "ListenAndServe" => FuncDecl
	idents      map[SpotKind]map[string][]Ident
	opts        indexOptions
}

func canonical(w string) string { return strings.ToLower(w) }

// Somewhat arbitrary, but I figure low enough to not hurt disk-based filesystems
// consuming file descriptors, where some systems have low 256 or 512 limits.
// Go should have a built-in way to cap fd usage under the ulimit.
const (
	maxOpenFiles = 200
	maxOpenDirs  = 50
)

func (c *Corpus) throttle() float64 {
	if c.IndexThrottle <= 0 {
		return 0.9
	}
	if c.IndexThrottle > 1.0 {
		return 1.0
	}
	return c.IndexThrottle
}

// NewIndex creates a new index for the .go files provided by the corpus.
func (c *Corpus) NewIndex() *Index {
	// initialize Indexer
	// (use some reasonably sized maps to start)
	x := &Indexer{
		c:           c,
		fset:        token.NewFileSet(),
		fsOpenGate:  make(chan bool, maxOpenFiles),
		strings:     make(map[string]string),
		packages:    make(map[Pak]*Pak, 256),
		words:       make(map[string]*IndexResult, 8192),
		throttle:    util.NewThrottle(c.throttle(), 100*time.Millisecond), // run at least 0.1s at a time
		importCount: make(map[string]int),
		packagePath: make(map[string]map[string]bool),
		exports:     make(map[string]map[string]SpotKind),
		idents:      make(map[SpotKind]map[string][]Ident, 4),
	}

	// index all files in the directories given by dirnames
	var wg sync.WaitGroup // outstanding ReadDir + visitFile
	dirGate := make(chan bool, maxOpenDirs)
	for dirname := range c.fsDirnames() {
		if c.IndexDirectory != nil && !c.IndexDirectory(dirname) {
			continue
		}
		dirGate <- true
		wg.Add(1)
		go func(dirname string) {
			defer func() { <-dirGate }()
			defer wg.Done()

			list, err := c.fs.ReadDir(dirname)
			if err != nil {
				log.Printf("ReadDir(%q): %v; skipping directory", dirname, err)
				return // ignore this directory
			}
			for _, fi := range list {
				wg.Add(1)
				go func(fi os.FileInfo) {
					defer wg.Done()
					x.visitFile(dirname, fi)
				}(fi)
			}
		}(dirname)
	}
	wg.Wait()

	if !c.IndexFullText {
		// the file set, the current file, and the sources are
		// not needed after indexing if no text index is built -
		// help GC and clear them
		x.fset = nil
		x.sources.Reset()
		x.current = nil // contains reference to fset!
	}

	// for each word, reduce the RunLists into a LookupResult;
	// also collect the word with its canonical spelling in a
	// word list for later computation of alternative spellings
	words := make(map[string]*LookupResult)
	var wlist RunList
	for w, h := range x.words {
		decls := reduce(h.Decls)
		others := reduce(h.Others)
		words[w] = &LookupResult{
			Decls:  decls,
			Others: others,
		}
		wlist = append(wlist, &wordPair{canonical(w), w})
		x.throttle.Throttle()
	}
	x.stats.Words = len(words)

	// reduce the word list {canonical(w), w} into
	// a list of AltWords runs {canonical(w), {w}}
	alist := wlist.reduce(lessWordPair, newAltWords)

	// convert alist into a map of alternative spellings
	alts := make(map[string]*AltWords)
	for i := 0; i < len(alist); i++ {
		a := alist[i].(*AltWords)
		alts[a.Canon] = a
	}

	// create text index
	var suffixes *suffixarray.Index
	if c.IndexFullText {
		suffixes = suffixarray.New(x.sources.Bytes())
	}

	// sort idents by the number of imports of their respective packages
	for _, idMap := range x.idents {
		for _, ir := range idMap {
			sort.Sort(byImportCount{ir, x.importCount})
		}
	}

	return &Index{
		fset:        x.fset,
		suffixes:    suffixes,
		words:       words,
		alts:        alts,
		snippets:    x.snippets,
		stats:       x.stats,
		importCount: x.importCount,
		packagePath: x.packagePath,
		exports:     x.exports,
		idents:      x.idents,
		opts: indexOptions{
			Docs:       x.c.IndexDocs,
			GoCode:     x.c.IndexGoCode,
			FullText:   x.c.IndexFullText,
			MaxResults: x.c.MaxResults,
		},
	}
}

var ErrFileIndexVersion = errors.New("file index version out of date")

const fileIndexVersion = 3

// fileIndex is the subset of Index that's gob-encoded for use by
// Index.Write and Index.Read.
type fileIndex struct {
	Version     int
	Words       map[string]*LookupResult
	Alts        map[string]*AltWords
	Snippets    []*Snippet
	Fulltext    bool
	Stats       Statistics
	ImportCount map[string]int
	PackagePath map[string]map[string]bool
	Exports     map[string]map[string]SpotKind
	Idents      map[SpotKind]map[string][]Ident
	Opts        indexOptions
}

func (x *fileIndex) Write(w io.Writer) error {
	return gob.NewEncoder(w).Encode(x)
}

func (x *fileIndex) Read(r io.Reader) error {
	return gob.NewDecoder(r).Decode(x)
}

// WriteTo writes the index x to w.
func (x *Index) WriteTo(w io.Writer) (n int64, err error) {
	w = countingWriter{&n, w}
	fulltext := false
	if x.suffixes != nil {
		fulltext = true
	}
	fx := fileIndex{
		Version:     fileIndexVersion,
		Words:       x.words,
		Alts:        x.alts,
		Snippets:    x.snippets,
		Fulltext:    fulltext,
		Stats:       x.stats,
		ImportCount: x.importCount,
		PackagePath: x.packagePath,
		Exports:     x.exports,
		Idents:      x.idents,
		Opts:        x.opts,
	}
	if err := fx.Write(w); err != nil {
		return 0, err
	}
	if fulltext {
		encode := func(x interface{}) error {
			return gob.NewEncoder(w).Encode(x)
		}
		if err := x.fset.Write(encode); err != nil {
			return 0, err
		}
		if err := x.suffixes.Write(w); err != nil {
			return 0, err
		}
	}
	return n, nil
}

// ReadFrom reads the index from r into x; x must not be nil.
// If r does not also implement io.ByteReader, it will be wrapped in a bufio.Reader.
// If the index is from an old version, the error is ErrFileIndexVersion.
func (x *Index) ReadFrom(r io.Reader) (n int64, err error) {
	// We use the ability to read bytes as a plausible surrogate for buffering.
	if _, ok := r.(io.ByteReader); !ok {
		r = bufio.NewReader(r)
	}
	r = countingReader{&n, r.(byteReader)}
	var fx fileIndex
	if err := fx.Read(r); err != nil {
		return n, err
	}
	if fx.Version != fileIndexVersion {
		return 0, ErrFileIndexVersion
	}
	x.words = fx.Words
	x.alts = fx.Alts
	x.snippets = fx.Snippets
	x.stats = fx.Stats
	x.importCount = fx.ImportCount
	x.packagePath = fx.PackagePath
	x.exports = fx.Exports
	x.idents = fx.Idents
	x.opts = fx.Opts
	if fx.Fulltext {
		x.fset = token.NewFileSet()
		decode := func(x interface{}) error {
			return gob.NewDecoder(r).Decode(x)
		}
		if err := x.fset.Read(decode); err != nil {
			return n, err
		}
		x.suffixes = new(suffixarray.Index)
		if err := x.suffixes.Read(r); err != nil {
			return n, err
		}
	}
	return n, nil
}

// Stats returns index statistics.
func (x *Index) Stats() Statistics {
	return x.stats
}

// ImportCount returns a map from import paths to how many times they were seen.
func (x *Index) ImportCount() map[string]int {
	return x.importCount
}

// PackagePath returns a map from short package name to a set
// of full package path names that use that short package name.
func (x *Index) PackagePath() map[string]map[string]bool {
	return x.packagePath
}

// Exports returns a map from full package path to exported
// symbol name to its type.
func (x *Index) Exports() map[string]map[string]SpotKind {
	return x.exports
}

// Idents returns a map from identifier type to exported
// symbol name to the list of identifiers matching that name.
func (x *Index) Idents() map[SpotKind]map[string][]Ident {
	return x.idents
}

func (x *Index) lookupWord(w string) (match *LookupResult, alt *AltWords) {
	match = x.words[w]
	alt = x.alts[canonical(w)]
	// remove current spelling from alternatives
	// (if there is no match, the alternatives do
	// not contain the current spelling)
	if match != nil && alt != nil {
		alt = alt.filter(w)
	}
	return
}

// isIdentifier reports whether s is a Go identifier.
func isIdentifier(s string) bool {
	for i, ch := range s {
		if unicode.IsLetter(ch) || ch == '_' || i > 0 && unicode.IsDigit(ch) {
			continue
		}
		return false
	}
	return len(s) > 0
}

// For a given query, which is either a single identifier or a qualified
// identifier, Lookup returns a SearchResult containing packages, a LookupResult, a
// list of alternative spellings, and identifiers, if any. Any and all results
// may be nil.  If the query syntax is wrong, an error is reported.
func (x *Index) Lookup(query string) (*SearchResult, error) {
	ss := strings.Split(query, ".")

	// check query syntax
	for _, s := range ss {
		if !isIdentifier(s) {
			return nil, errors.New("all query parts must be identifiers")
		}
	}
	rslt := &SearchResult{
		Query:  query,
		Idents: make(map[SpotKind][]Ident, 5),
	}
	// handle simple and qualified identifiers
	switch len(ss) {
	case 1:
		ident := ss[0]
		rslt.Hit, rslt.Alt = x.lookupWord(ident)
		if rslt.Hit != nil {
			// found a match - filter packages with same name
			// for the list of packages called ident, if any
			rslt.Pak = rslt.Hit.Others.filter(ident)
		}
		for k, v := range x.idents {
			const rsltLimit = 50
			ids := byImportCount{v[ident], x.importCount}
			rslt.Idents[k] = ids.top(rsltLimit)
		}

	case 2:
		pakname, ident := ss[0], ss[1]
		rslt.Hit, rslt.Alt = x.lookupWord(ident)
		if rslt.Hit != nil {
			// found a match - filter by package name
			// (no paks - package names are not qualified)
			decls := rslt.Hit.Decls.filter(pakname)
			others := rslt.Hit.Others.filter(pakname)
			rslt.Hit = &LookupResult{decls, others}
		}
		for k, v := range x.idents {
			ids := byImportCount{v[ident], x.importCount}
			rslt.Idents[k] = ids.filter(pakname)
		}

	default:
		return nil, errors.New("query is not a (qualified) identifier")
	}

	return rslt, nil
}

func (x *Index) Snippet(i int) *Snippet {
	// handle illegal snippet indices gracefully
	if 0 <= i && i < len(x.snippets) {
		return x.snippets[i]
	}
	return nil
}

type positionList []struct {
	filename string
	line     int
}

func (list positionList) Len() int           { return len(list) }
func (list positionList) Less(i, j int) bool { return list[i].filename < list[j].filename }
func (list positionList) Swap(i, j int)      { list[i], list[j] = list[j], list[i] }

// unique returns the list sorted and with duplicate entries removed
func unique(list []int) []int {
	sort.Ints(list)
	var last int
	i := 0
	for _, x := range list {
		if i == 0 || x != last {
			last = x
			list[i] = x
			i++
		}
	}
	return list[0:i]
}

// A FileLines value specifies a file and line numbers within that file.
type FileLines struct {
	Filename string
	Lines    []int
}

// LookupRegexp returns the number of matches and the matches where a regular
// expression r is found in the full text index. At most n matches are
// returned (thus found <= n).
func (x *Index) LookupRegexp(r *regexp.Regexp, n int) (found int, result []FileLines) {
	if x.suffixes == nil || n <= 0 {
		return
	}
	// n > 0

	var list positionList
	// FindAllIndex may returns matches that span across file boundaries.
	// Such matches are unlikely, buf after eliminating them we may end up
	// with fewer than n matches. If we don't have enough at the end, redo
	// the search with an increased value n1, but only if FindAllIndex
	// returned all the requested matches in the first place (if it
	// returned fewer than that there cannot be more).
	for n1 := n; found < n; n1 += n - found {
		found = 0
		matches := x.suffixes.FindAllIndex(r, n1)
		// compute files, exclude matches that span file boundaries,
		// and map offsets to file-local offsets
		list = make(positionList, len(matches))
		for _, m := range matches {
			// by construction, an offset corresponds to the Pos value
			// for the file set - use it to get the file and line
			p := token.Pos(m[0])
			if file := x.fset.File(p); file != nil {
				if base := file.Base(); base <= m[1] && m[1] <= base+file.Size() {
					// match [m[0], m[1]) is within the file boundaries
					list[found].filename = file.Name()
					list[found].line = file.Line(p)
					found++
				}
			}
		}
		if found == n || len(matches) < n1 {
			// found all matches or there's no chance to find more
			break
		}
	}
	list = list[0:found]
	sort.Sort(list) // sort by filename

	// collect matches belonging to the same file
	var last string
	var lines []int
	addLines := func() {
		if len(lines) > 0 {
			// remove duplicate lines
			result = append(result, FileLines{last, unique(lines)})
			lines = nil
		}
	}
	for _, m := range list {
		if m.filename != last {
			addLines()
			last = m.filename
		}
		lines = append(lines, m.line)
	}
	addLines()

	return
}

// invalidateIndex should be called whenever any of the file systems
// under godoc's observation change so that the indexer is kicked on.
func (c *Corpus) invalidateIndex() {
	c.fsModified.Set(nil)
	c.refreshMetadata()
}

// feedDirnames feeds the directory names of all directories
// under the file system given by root to channel c.
func (c *Corpus) feedDirnames(ch chan<- string) {
	if dir, _ := c.fsTree.Get(); dir != nil {
		for d := range dir.(*Directory).iter(false) {
			ch <- d.Path
		}
	}
}

// fsDirnames() returns a channel sending all directory names
// of all the file systems under godoc's observation.
func (c *Corpus) fsDirnames() <-chan string {
	ch := make(chan string, 256) // buffered for fewer context switches
	go func() {
		c.feedDirnames(ch)
		close(ch)
	}()
	return ch
}

// CompatibleWith reports whether the Index x is compatible with the corpus
// indexing options set in c.
func (x *Index) CompatibleWith(c *Corpus) bool {
	return x.opts.Docs == c.IndexDocs &&
		x.opts.GoCode == c.IndexGoCode &&
		x.opts.FullText == c.IndexFullText &&
		x.opts.MaxResults == c.MaxResults
}

func (c *Corpus) readIndex(filenames string) error {
	matches, err := filepath.Glob(filenames)
	if err != nil {
		return err
	} else if matches == nil {
		return fmt.Errorf("no index files match %q", filenames)
	}
	sort.Strings(matches) // make sure files are in the right order
	files := make([]io.Reader, 0, len(matches))
	for _, filename := range matches {
		f, err := os.Open(filename)
		if err != nil {
			return err
		}
		defer f.Close()
		files = append(files, f)
	}
	return c.ReadIndexFrom(io.MultiReader(files...))
}

// ReadIndexFrom sets the current index from the serialized version found in r.
func (c *Corpus) ReadIndexFrom(r io.Reader) error {
	x := new(Index)
	if _, err := x.ReadFrom(r); err != nil {
		return err
	}
	if !x.CompatibleWith(c) {
		return fmt.Errorf("index file options are incompatible: %v", x.opts)
	}
	c.searchIndex.Set(x)
	return nil
}

func (c *Corpus) UpdateIndex() {
	if c.Verbose {
		log.Printf("updating index...")
	}
	start := time.Now()
	index := c.NewIndex()
	stop := time.Now()
	c.searchIndex.Set(index)
	if c.Verbose {
		secs := stop.Sub(start).Seconds()
		stats := index.Stats()
		log.Printf("index updated (%gs, %d bytes of source, %d files, %d lines, %d unique words, %d spots)",
			secs, stats.Bytes, stats.Files, stats.Lines, stats.Words, stats.Spots)
	}
	memstats := new(runtime.MemStats)
	runtime.ReadMemStats(memstats)
	if c.Verbose {
		log.Printf("before GC: bytes = %d footprint = %d", memstats.HeapAlloc, memstats.Sys)
	}
	runtime.GC()
	runtime.ReadMemStats(memstats)
	if c.Verbose {
		log.Printf("after  GC: bytes = %d footprint = %d", memstats.HeapAlloc, memstats.Sys)
	}
}

// RunIndexer runs forever, indexing.
func (c *Corpus) RunIndexer() {
	// initialize the index from disk if possible
	if c.IndexFiles != "" {
		c.initFSTree()
		if err := c.readIndex(c.IndexFiles); err != nil {
			log.Printf("error reading index from file %s: %v", c.IndexFiles, err)
		}
		return
	}

	// Repeatedly update the package directory tree and index.
	for {
		c.initFSTree()
		c.UpdateIndex()
		if c.IndexInterval < 0 {
			return
		}
		delay := 5 * time.Minute // by default, reindex every 5 minutes
		if c.IndexInterval > 0 {
			delay = c.IndexInterval
		}
		time.Sleep(delay)
	}
}

type countingWriter struct {
	n *int64
	w io.Writer
}

func (c countingWriter) Write(p []byte) (n int, err error) {
	n, err = c.w.Write(p)
	*c.n += int64(n)
	return
}

type byteReader interface {
	io.Reader
	io.ByteReader
}

type countingReader struct {
	n *int64
	r byteReader
}

func (c countingReader) Read(p []byte) (n int, err error) {
	n, err = c.r.Read(p)
	*c.n += int64(n)
	return
}

func (c countingReader) ReadByte() (b byte, err error) {
	b, err = c.r.ReadByte()
	*c.n += 1
	return
}
