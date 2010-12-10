// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains the infrastructure to create an
// identifier and full-text index for a set of Go files.
//
// Algorithm for identifier index:
// - traverse all .go files of the file tree specified by root
// - for each word (identifier) encountered, collect all occurences (spots)
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

package main

import (
	"bytes"
	"container/vector"
	"go/ast"
	"go/parser"
	"go/token"
	"go/scanner"
	"index/suffixarray"
	"io/ioutil"
	"os"
	pathutil "path"
	"sort"
	"strings"
)


// ----------------------------------------------------------------------------
// RunList

// A RunList is a vector of entries that can be sorted according to some
// criteria. A RunList may be compressed by grouping "runs" of entries
// which are equal (according to the sort critera) into a new RunList of
// runs. For instance, a RunList containing pairs (x, y) may be compressed
// into a RunList containing pair runs (x, {y}) where each run consists of
// a list of y's with the same x.
type RunList struct {
	vector.Vector
	less func(x, y interface{}) bool
}

func (h *RunList) Less(i, j int) bool { return h.less(h.At(i), h.At(j)) }


func (h *RunList) sort(less func(x, y interface{}) bool) {
	h.less = less
	sort.Sort(h)
}


// Compress entries which are the same according to a sort criteria
// (specified by less) into "runs".
func (h *RunList) reduce(less func(x, y interface{}) bool, newRun func(h *RunList, i, j int) interface{}) *RunList {
	// create runs of entries with equal values
	h.sort(less)

	// for each run, make a new run object and collect them in a new RunList
	var hh RunList
	i := 0
	for j := 0; j < h.Len(); j++ {
		if less(h.At(i), h.At(j)) {
			hh.Push(newRun(h, i, j))
			i = j // start a new run
		}
	}
	// add final run, if any
	if i < h.Len() {
		hh.Push(newRun(h, i, h.Len()))
	}

	return &hh
}


// ----------------------------------------------------------------------------
// SpotInfo

// A SpotInfo value describes a particular identifier spot in a given file;
// It encodes three values: the SpotKind (declaration or use), a line or
// snippet index "lori", and whether it's a line or index.
//
// The following encoding is used:
//
//   bits    32   4    1       0
//   value    [lori|kind|isIndex]
//
type SpotInfo uint32

// SpotKind describes whether an identifier is declared (and what kind of
// declaration) or used.
type SpotKind uint32

const (
	PackageClause SpotKind = iota
	ImportDecl
	ConstDecl
	TypeDecl
	VarDecl
	FuncDecl
	MethodDecl
	Use
	nKinds
)


func init() {
	// sanity check: if nKinds is too large, the SpotInfo
	// accessor functions may need to be updated
	if nKinds > 8 {
		panic("nKinds > 8")
	}
}


// makeSpotInfo makes a SpotInfo.
func makeSpotInfo(kind SpotKind, lori int, isIndex bool) SpotInfo {
	// encode lori: bits [4..32)
	x := SpotInfo(lori) << 4
	if int(x>>4) != lori {
		// lori value doesn't fit - since snippet indices are
		// most certainly always smaller then 1<<28, this can
		// only happen for line numbers; give it no line number (= 0)
		x = 0
	}
	// encode kind: bits [1..4)
	x |= SpotInfo(kind) << 1
	// encode isIndex: bit 0
	if isIndex {
		x |= 1
	}
	return x
}


func (x SpotInfo) Kind() SpotKind { return SpotKind(x >> 1 & 7) }
func (x SpotInfo) Lori() int      { return int(x >> 4) }
func (x SpotInfo) IsIndex() bool  { return x&1 != 0 }


// ----------------------------------------------------------------------------
// KindRun

// Debugging support. Disable to see multiple entries per line.
const removeDuplicates = true

// A KindRun is a run of SpotInfos of the same kind in a given file.
type KindRun struct {
	Kind  SpotKind
	Infos []SpotInfo
}


// KindRuns are sorted by line number or index. Since the isIndex bit
// is always the same for all infos in one list we can compare lori's.
func (f *KindRun) Len() int           { return len(f.Infos) }
func (f *KindRun) Less(i, j int) bool { return f.Infos[i].Lori() < f.Infos[j].Lori() }
func (f *KindRun) Swap(i, j int)      { f.Infos[i], f.Infos[j] = f.Infos[j], f.Infos[i] }


// FileRun contents are sorted by Kind for the reduction into KindRuns.
func lessKind(x, y interface{}) bool { return x.(SpotInfo).Kind() < y.(SpotInfo).Kind() }


// newKindRun allocates a new KindRun from the SpotInfo run [i, j) in h.
func newKindRun(h *RunList, i, j int) interface{} {
	kind := h.At(i).(SpotInfo).Kind()
	infos := make([]SpotInfo, j-i)
	k := 0
	for ; i < j; i++ {
		infos[k] = h.At(i).(SpotInfo)
		k++
	}
	run := &KindRun{kind, infos}

	// Spots were sorted by file and kind to create this run.
	// Within this run, sort them by line number or index.
	sort.Sort(run)

	if removeDuplicates {
		// Since both the lori and kind field must be
		// same for duplicates, and since the isIndex
		// bit is always the same for all infos in one
		// list we can simply compare the entire info.
		k := 0
		var prev SpotInfo
		for i, x := range infos {
			if x != prev || i == 0 {
				infos[k] = x
				k++
				prev = x
			}
		}
		run.Infos = infos[0:k]
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
	Path string // complete file name
	Pak  Pak    // the package to which the file belongs
}


// A Spot describes a single occurence of a word.
type Spot struct {
	File *File
	Info SpotInfo
}


// A FileRun is a list of KindRuns belonging to the same file.
type FileRun struct {
	File   *File
	Groups []*KindRun
}


// Spots are sorted by path for the reduction into FileRuns.
func lessSpot(x, y interface{}) bool { return x.(Spot).File.Path < y.(Spot).File.Path }


// newFileRun allocates a new FileRun from the Spot run [i, j) in h.
func newFileRun(h0 *RunList, i, j int) interface{} {
	file := h0.At(i).(Spot).File

	// reduce the list of Spots into a list of KindRuns
	var h1 RunList
	h1.Vector.Resize(j-i, 0)
	k := 0
	for ; i < j; i++ {
		h1.Set(k, h0.At(i).(Spot).Info)
		k++
	}
	h2 := h1.reduce(lessKind, newKindRun)

	// create the FileRun
	groups := make([]*KindRun, h2.Len())
	for i := 0; i < h2.Len(); i++ {
		groups[i] = h2.At(i).(*KindRun)
	}
	return &FileRun{file, groups}
}


// ----------------------------------------------------------------------------
// PakRun

// A PakRun describes a run of *FileRuns of a package.
type PakRun struct {
	Pak   Pak
	Files []*FileRun
}

// Sorting support for files within a PakRun.
func (p *PakRun) Len() int           { return len(p.Files) }
func (p *PakRun) Less(i, j int) bool { return p.Files[i].File.Path < p.Files[j].File.Path }
func (p *PakRun) Swap(i, j int)      { p.Files[i], p.Files[j] = p.Files[j], p.Files[i] }


// FileRuns are sorted by package for the reduction into PakRuns.
func lessFileRun(x, y interface{}) bool {
	return x.(*FileRun).File.Pak.less(&y.(*FileRun).File.Pak)
}


// newPakRun allocates a new PakRun from the *FileRun run [i, j) in h.
func newPakRun(h *RunList, i, j int) interface{} {
	pak := h.At(i).(*FileRun).File.Pak
	files := make([]*FileRun, j-i)
	k := 0
	for ; i < j; i++ {
		files[k] = h.At(i).(*FileRun)
		k++
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
func lessPakRun(x, y interface{}) bool { return x.(*PakRun).Pak.less(&y.(*PakRun).Pak) }


func reduce(h0 *RunList) HitList {
	// reduce a list of Spots into a list of FileRuns
	h1 := h0.reduce(lessSpot, newFileRun)
	// reduce a list of FileRuns into a list of PakRuns
	h2 := h1.reduce(lessFileRun, newPakRun)
	// sort the list of PakRuns by package
	h2.sort(lessPakRun)
	// create a HitList
	h := make(HitList, h2.Len())
	for i := 0; i < h2.Len(); i++ {
		h[i] = h2.At(i).(*PakRun)
	}
	return h
}


func (h HitList) filter(pakname string) HitList {
	// determine number of matching packages (most of the time just one)
	n := 0
	for _, p := range h {
		if p.Pak.Name == pakname {
			n++
		}
	}
	// create filtered HitList
	hh := make(HitList, n)
	i := 0
	for _, p := range h {
		if p.Pak.Name == pakname {
			hh[i] = p
			i++
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


// newAltWords allocates a new AltWords from the *wordPair run [i, j) in h.
func newAltWords(h *RunList, i, j int) interface{} {
	canon := h.At(i).(*wordPair).canon
	alts := make([]string, j-i)
	k := 0
	for ; i < j; i++ {
		alts[k] = h.At(i).(*wordPair).alt
		k++
	}
	return &AltWords{canon, alts}
}


func (a *AltWords) filter(s string) *AltWords {
	if len(a.Alts) == 1 && a.Alts[0] == s {
		// there are no different alternatives
		return nil
	}

	// make a new AltWords with the current spelling removed
	alts := make([]string, len(a.Alts))
	i := 0
	for _, w := range a.Alts {
		if w != s {
			alts[i] = w
			i++
		}
	}
	return &AltWords{a.Canon, alts[0:i]}
}


// ----------------------------------------------------------------------------
// Indexer

// Adjust these flags as seems best.
const excludeMainPackages = false
const excludeTestFiles = false


type IndexResult struct {
	Decls  RunList // package-level declarations (with snippets)
	Others RunList // all other occurences
}


// Statistics provides statistics information for an index.
type Statistics struct {
	Bytes int // total size of indexed source files
	Files int // number of indexed source files
	Words int // number of different identifiers
	Spots int // number of identifier occurences
}


// An Indexer maintains the data structures and provides the machinery
// for indexing .go files under a file tree. It implements the path.Visitor
// interface for walking file trees, and the ast.Visitor interface for
// walking Go ASTs.
type Indexer struct {
	fset     *token.FileSet          // file set for all indexed files
	sources  bytes.Buffer            // concatenated sources
	words    map[string]*IndexResult // RunLists of Spots
	snippets vector.Vector           // vector of *Snippets, indexed by snippet indices
	current  *token.File             // last file added to file set
	file     *File                   // AST for current file
	decl     ast.Decl                // AST for current decl
	stats    Statistics
}


func (x *Indexer) addSnippet(s *Snippet) int {
	index := x.snippets.Len()
	x.snippets.Push(s)
	return index
}


func (x *Indexer) visitComment(c *ast.CommentGroup) {
	if c != nil {
		ast.Walk(x, c)
	}
}


func (x *Indexer) visitIdent(kind SpotKind, id *ast.Ident) {
	if id != nil {
		lists, found := x.words[id.Name]
		if !found {
			lists = new(IndexResult)
			x.words[id.Name] = lists
		}

		if kind == Use || x.decl == nil {
			// not a declaration or no snippet required
			info := makeSpotInfo(kind, x.current.Line(id.Pos()), false)
			lists.Others.Push(Spot{x.file, info})
		} else {
			// a declaration with snippet
			index := x.addSnippet(NewSnippet(x.fset, x.decl, id))
			info := makeSpotInfo(kind, index, true)
			lists.Decls.Push(Spot{x.file, info})
		}

		x.stats.Spots++
	}
}


func (x *Indexer) visitSpec(spec ast.Spec, isVarDecl bool) {
	switch n := spec.(type) {
	case *ast.ImportSpec:
		x.visitComment(n.Doc)
		x.visitIdent(ImportDecl, n.Name)
		ast.Walk(x, n.Path)
		x.visitComment(n.Comment)

	case *ast.ValueSpec:
		x.visitComment(n.Doc)
		kind := ConstDecl
		if isVarDecl {
			kind = VarDecl
		}
		for _, n := range n.Names {
			x.visitIdent(kind, n)
		}
		ast.Walk(x, n.Type)
		for _, v := range n.Values {
			ast.Walk(x, v)
		}
		x.visitComment(n.Comment)

	case *ast.TypeSpec:
		x.visitComment(n.Doc)
		x.visitIdent(TypeDecl, n.Name)
		ast.Walk(x, n.Type)
		x.visitComment(n.Comment)
	}
}


func (x *Indexer) Visit(node ast.Node) ast.Visitor {
	// TODO(gri): methods in interface types are categorized as VarDecl
	switch n := node.(type) {
	case nil:
		return nil

	case *ast.Ident:
		x.visitIdent(Use, n)

	case *ast.Field:
		x.decl = nil // no snippets for fields
		x.visitComment(n.Doc)
		for _, m := range n.Names {
			x.visitIdent(VarDecl, m)
		}
		ast.Walk(x, n.Type)
		ast.Walk(x, n.Tag)
		x.visitComment(n.Comment)

	case *ast.DeclStmt:
		if decl, ok := n.Decl.(*ast.GenDecl); ok {
			// local declarations can only be *ast.GenDecls
			x.decl = nil // no snippets for local declarations
			x.visitComment(decl.Doc)
			for _, s := range decl.Specs {
				x.visitSpec(s, decl.Tok == token.VAR)
			}
		} else {
			// handle error case gracefully
			ast.Walk(x, n.Decl)
		}

	case *ast.GenDecl:
		x.decl = n
		x.visitComment(n.Doc)
		for _, s := range n.Specs {
			x.visitSpec(s, n.Tok == token.VAR)
		}

	case *ast.FuncDecl:
		x.visitComment(n.Doc)
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
		x.visitComment(n.Doc)
		x.decl = nil
		x.visitIdent(PackageClause, n.Name)
		for _, d := range n.Decls {
			ast.Walk(x, d)
		}
		// don't visit package level comments for now
		// to avoid duplicate visiting from individual
		// nodes

	default:
		return x
	}

	return nil
}


func pkgName(filename string) string {
	// use a new file set each time in order to not pollute the indexer's
	// file set (which must stay in sync with the concatenated source code)
	file, err := parser.ParseFile(token.NewFileSet(), filename, nil, parser.PackageClauseOnly)
	if err != nil || file == nil {
		return ""
	}
	return file.Name.Name
}


func (x *Indexer) addFile(filename string) *ast.File {
	// open file
	f, err := os.Open(filename, os.O_RDONLY, 0)
	if err != nil {
		return nil
	}
	defer f.Close()

	// The file set's base offset and x.sources size must be in lock-step;
	// this permits the direct mapping of suffix array lookup results to
	// to corresponding Pos values.
	//
	// When a file is added to the file set, it's offset base increases by
	// the size of the file + 1; and the initial base offset is 1. Add an
	// extra byte to the sources here.
	x.sources.WriteByte(0)

	// If the sources length doesn't match the file set base at this point
	// the file set implementation changed or we have another error.
	base := x.fset.Base()
	if x.sources.Len() != base {
		panic("internal error - file base incorrect")
	}

	// append file contents to x.sources
	if _, err := x.sources.ReadFrom(f); err != nil {
		x.sources.Truncate(base) // discard possibly added data
		return nil               // ignore files with I/O errors
	}

	// parse the file and in the process add it to the file set
	src := x.sources.Bytes()[base:] // no need to reread the file
	file, err := parser.ParseFile(x.fset, filename, src, parser.ParseComments)
	if err != nil {
		// do not discard the added source code in this case
		// because the file has been added to the file set and
		// the source size must match the file set base
		// TODO(gri): given a FileSet.RemoveFile() one might be
		//            able to discard the data here (worthwhile?)
		return nil // ignore files with (parse) errors
	}

	return file
}


func (x *Indexer) visitFile(dirname string, f *os.FileInfo) {
	// for now, exclude bug257.go as it causes problems with suffixarray
	// TODO fix index/suffixarray
	if f.Name == "bug257.go" {
		return
	}

	if !isGoFile(f) {
		return
	}

	path := pathutil.Join(dirname, f.Name)
	if excludeTestFiles && (!isPkgFile(f) || strings.HasPrefix(path, "test/")) {
		return
	}

	if excludeMainPackages && pkgName(path) == "main" {
		return
	}

	file := x.addFile(path)
	if file == nil {
		return
	}

	// we've got a file to index
	x.current = x.fset.File(file.Pos()) // file.Pos is in the current file
	dir, _ := pathutil.Split(path)
	pak := Pak{dir, file.Name.Name}
	x.file = &File{path, pak}
	ast.Walk(x, file)

	// update statistics
	// (count real file size as opposed to using the padded x.sources.Len())
	x.stats.Bytes += x.current.Size()
	x.stats.Files++
}


// ----------------------------------------------------------------------------
// Index

type LookupResult struct {
	Decls  HitList // package-level declarations (with snippets)
	Others HitList // all other occurences
}


type Index struct {
	fset     *token.FileSet           // file set used during indexing; nil if no textindex
	suffixes *suffixarray.Index       // suffixes for concatenated sources; nil if no textindex
	words    map[string]*LookupResult // maps words to hit lists
	alts     map[string]*AltWords     // maps canonical(words) to lists of alternative spellings
	snippets []*Snippet               // all snippets, indexed by snippet index
	stats    Statistics
}


func canonical(w string) string { return strings.ToLower(w) }


// NewIndex creates a new index for the .go files
// in the directories given by dirnames.
//
func NewIndex(dirnames <-chan string, fulltextIndex bool) *Index {
	var x Indexer

	// initialize Indexer
	x.fset = token.NewFileSet()
	x.words = make(map[string]*IndexResult)

	// index all files in the directories given by dirnames
	for dirname := range dirnames {
		list, err := ioutil.ReadDir(dirname)
		if err != nil {
			continue // ignore this directory
		}
		for _, f := range list {
			if !f.IsDirectory() {
				x.visitFile(dirname, f)
			}
		}
	}

	if !fulltextIndex {
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
		decls := reduce(&h.Decls)
		others := reduce(&h.Others)
		words[w] = &LookupResult{
			Decls:  decls,
			Others: others,
		}
		wlist.Push(&wordPair{canonical(w), w})
	}
	x.stats.Words = len(words)

	// reduce the word list {canonical(w), w} into
	// a list of AltWords runs {canonical(w), {w}}
	alist := wlist.reduce(lessWordPair, newAltWords)

	// convert alist into a map of alternative spellings
	alts := make(map[string]*AltWords)
	for i := 0; i < alist.Len(); i++ {
		a := alist.At(i).(*AltWords)
		alts[a.Canon] = a
	}

	// convert snippet vector into a list
	snippets := make([]*Snippet, x.snippets.Len())
	for i := 0; i < x.snippets.Len(); i++ {
		snippets[i] = x.snippets.At(i).(*Snippet)
	}

	// create text index
	var suffixes *suffixarray.Index
	if fulltextIndex {
		suffixes = suffixarray.New(x.sources.Bytes())
	}

	return &Index{x.fset, suffixes, words, alts, snippets, x.stats}
}


// Stats() returns index statistics.
func (x *Index) Stats() Statistics {
	return x.stats
}


func (x *Index) LookupWord(w string) (match *LookupResult, alt *AltWords) {
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


func isIdentifier(s string) bool {
	var S scanner.Scanner
	S.Init(token.NewFileSet(), "", []byte(s), nil, 0)
	if _, tok, _ := S.Scan(); tok == token.IDENT {
		_, tok, _ := S.Scan()
		return tok == token.EOF
	}
	return false
}


// For a given query, which is either a single identifier or a qualified
// identifier, Lookup returns a LookupResult, and a list of alternative
// spellings, if any. If the query syntax is wrong, illegal is set.
func (x *Index) Lookup(query string) (match *LookupResult, alt *AltWords, illegal bool) {
	ss := strings.Split(query, ".", -1)

	// check query syntax
	for _, s := range ss {
		if !isIdentifier(s) {
			illegal = true
			return
		}
	}

	switch len(ss) {
	case 1:
		match, alt = x.LookupWord(ss[0])

	case 2:
		pakname := ss[0]
		match, alt = x.LookupWord(ss[1])
		if match != nil {
			// found a match - filter by package name
			decls := match.Decls.filter(pakname)
			others := match.Others.filter(pakname)
			match = &LookupResult{decls, others}
		}

	default:
		illegal = true
	}

	return
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


// A Positions value specifies a file and line numbers within that file.
type Positions struct {
	Filename string
	Lines    []int
}


// LookupString returns a list of positions where a string s is found
// in the full text index and whether the result is complete or not.
// At most n positions (filename and line) are returned. The result is
// not complete if the index is not present or there are more than n
// occurrences of s.
//
func (x *Index) LookupString(s string, n int) (result []Positions, complete bool) {
	if x.suffixes == nil {
		return
	}

	offsets := x.suffixes.Lookup([]byte(s), n+1)
	if len(offsets) <= n {
		complete = true
	} else {
		offsets = offsets[0:n]
	}

	// compute file names and lines and sort the list by filename
	list := make(positionList, len(offsets))
	for i, offs := range offsets {
		// by construction, an offs corresponds to
		// the Pos value for the file set - use it
		// to get full Position information
		pos := x.fset.Position(token.Pos(offs))
		list[i].filename = pos.Filename
		list[i].line = pos.Line
	}
	sort.Sort(list)

	// compact positions with equal file names
	var last string
	var lines []int
	for _, pos := range list {
		if pos.filename != last {
			if len(lines) > 0 {
				result = append(result, Positions{last, lines})
				lines = nil
			}
			last = pos.filename
		}
		lines = append(lines, pos.line)
	}
	if len(lines) > 0 {
		result = append(result, Positions{last, lines})
	}

	return
}
