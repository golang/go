// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains the infrastructure to create an
// (identifier) index for a set of Go files.
//
// Basic indexing algorithm:
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

package main

import (
	"container/vector";
	"go/ast";
	"go/parser";
	"go/token";
	"os";
	pathutil "path";
	"sort";
	"strings";
)


// ----------------------------------------------------------------------------
// Data structures used during indexing

// A RunList is a vector of entries that can be sorted according to some
// criteria. A RunList may be compressed by grouping "runs" of entries
// which are equal (according to the sort critera) into a new RunList of
// runs. For instance, a RunList containing pairs (x, y) may be compressed
// into a RunList containing pair runs (x, {y}) where each run consists of
// a list of y's with the same x.
type RunList struct {
	vector.Vector;
	less	func(x, y interface{}) bool;
}

func (h *RunList) Less(i, j int) bool	{ return h.less(h.At(i), h.At(j)) }


func (h *RunList) sort(less func(x, y interface{}) bool) {
	h.less = less;
	sort.Sort(h);
}


// Compress entries which are the same according to a sort criteria
// (specified by less) into "runs".
func (h *RunList) reduce(less func(x, y interface{}) bool, newRun func(h *RunList, i, j int) interface{}) *RunList {
	// create runs of entries with equal values
	h.sort(less);

	// for each run, make a new run object and collect them in a new RunList
	var hh RunList;
	i := 0;
	for j := 0; j < h.Len(); j++ {
		if less(h.At(i), h.At(j)) {
			hh.Push(newRun(h, i, j));
			i = j;	// start a new run
		}
	}
	// add final run, if any
	if i < h.Len() {
		hh.Push(newRun(h, i, h.Len()));
	}

	return &hh;
}


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
	PackageClause	SpotKind	= iota;
	ImportDecl;
	ConstDecl;
	TypeDecl;
	VarDecl;
	FuncDecl;
	MethodDecl;
	Use;
	nKinds;
)


func init() {
	// sanity check: if nKinds is too large, the SpotInfo
	// accessor functions may need to be updated
	if nKinds > 8 {
		panic();
	}
}


// makeSpotInfo makes a SpotInfo.
func makeSpotInfo(kind SpotKind, lori int, isIndex bool) SpotInfo {
	// encode lori: bits [4..32)
	x := SpotInfo(lori)<<4;
	if int(x>>4) != lori {
		// lori value doesn't fit - since snippet indices are
		// most certainly always smaller then 1<<28, this can
		// only happen for line numbers; give it no line number (= 0)
		x = 0;
	}
	// encode kind: bits [1..4)
	x |= SpotInfo(kind)<<1;
	// encode isIndex: bit 0
	if isIndex {
		x |= 1;
	}
	return x;
}


func (x SpotInfo) less(y SpotInfo) bool	{ return x.Lori() < y.Lori() }


func (x SpotInfo) Kind() SpotKind	{ return SpotKind(x>>1&7) }


func (x SpotInfo) Lori() int	{ return int(x>>4) }


func (x SpotInfo) IsIndex() bool	{ return x&1 != 0 }


// A Pak describes a Go package.
type Pak struct {
	Path	string;	// directory name containing the package
	Name	string;	// package name as declared by package clause
}


// Paks are sorted by name (primary key) and by import path (secondary key).
func (p *Pak) less(q *Pak) bool {
	return p.Name < q.Name || p.Name == q.Name && p.Path < q.Path;
}


// A File describes a Go file.
type File struct {
	Path	string;	// complete file name
	Pak	Pak;	// the package to which the file belongs
}


func (f *File) less(g *File) bool	{ return f.Path < g.Path }


// A Spot describes a single occurence of a word.
type Spot struct {
	File	*File;
	Info	SpotInfo;
}


// Spots are sorted by filename.
func lessSpot(x, y interface{}) bool	{ return x.(Spot).File.less(y.(Spot).File) }


// A FileRun describes a run of Spots of a word in a single file.
type FileRun struct {
	File	*File;
	Infos	[]SpotInfo;
}


func (f *FileRun) Len() int		{ return len(f.Infos) }
func (f *FileRun) Less(i, j int) bool	{ return f.Infos[i].less(f.Infos[j]) }
func (f *FileRun) Swap(i, j int)	{ f.Infos[i], f.Infos[j] = f.Infos[j], f.Infos[i] }


// newFileRun allocates a new *FileRun from the Spot run [i, j) in h.
func newFileRun(h *RunList, i, j int) interface{} {
	file := h.At(i).(Spot).File;
	infos := make([]SpotInfo, j-i);
	k := 0;
	for ; i < j; i++ {
		infos[k] = h.At(i).(Spot).Info;
		k++;
	}
	run := &FileRun{file, infos};
	// Spots were sorted by file to create this run.
	// Within this run, sort them by line number.
	sort.Sort(run);
	// Remove duplicates: Both the lori and kind field
	// must be the same for duplicate, and since the
	// isIndex field is always the same for all infos
	// in one list we can simply compare the entire
	// info.
	k = 0;
	var prev SpotInfo;
	for i, x := range infos {
		if x != prev || i == 0 {
			infos[k] = x;
			k++;
			prev = x;
		}
	}
	run.Infos = infos[0:k];
	return run;
}


// FileRuns are sorted by package.
func lessFileRun(x, y interface{}) bool {
	return x.(*FileRun).File.Pak.less(&y.(*FileRun).File.Pak);
}


// A PakRun describes a run of *FileRuns of a package.
type PakRun struct {
	Pak	Pak;
	Files	[]*FileRun;
}

// Sorting support for files within a PakRun.
func (p *PakRun) Len() int		{ return len(p.Files) }
func (p *PakRun) Less(i, j int) bool	{ return p.Files[i].File.less(p.Files[j].File) }
func (p *PakRun) Swap(i, j int)		{ p.Files[i], p.Files[j] = p.Files[j], p.Files[i] }


// newPakRun allocates a new *PakRun from the *FileRun run [i, j) in h.
func newPakRun(h *RunList, i, j int) interface{} {
	pak := h.At(i).(*FileRun).File.Pak;
	files := make([]*FileRun, j-i);
	k := 0;
	for ; i < j; i++ {
		files[k] = h.At(i).(*FileRun);
		k++;
	}
	run := &PakRun{pak, files};
	sort.Sort(run);	// files were sorted by package; sort them by file now
	return run;
}


// PakRuns are sorted by package.
func lessPakRun(x, y interface{}) bool	{ return x.(*PakRun).Pak.less(&y.(*PakRun).Pak) }


// A HitList describes a list of PakRuns.
type HitList []*PakRun


func reduce(h0 *RunList) HitList {
	// reduce a list of Spots into a list of FileRuns
	h1 := h0.reduce(lessSpot, newFileRun);
	// reduce a list of FileRuns into a list of PakRuns
	h2 := h1.reduce(lessFileRun, newPakRun);
	// sort the list of PakRuns by package
	h2.sort(lessPakRun);
	// create a HitList
	h := make(HitList, h2.Len());
	for i := 0; i < h2.Len(); i++ {
		h[i] = h2.At(i).(*PakRun);
	}
	return h;
}


func (h HitList) filter(pakname string) HitList {
	// determine number of matching packages (most of the time just one)
	n := 0;
	for _, p := range h {
		if p.Pak.Name == pakname {
			n++;
		}
	}
	// create filtered HitList
	hh := make(HitList, n);
	i := 0;
	for _, p := range h {
		if p.Pak.Name == pakname {
			hh[i] = p;
			i++;
		}
	}
	return hh;
}


type wordPair struct {
	canon	string;	// canonical word spelling (all lowercase)
	alt	string;	// alternative spelling
}


// An AltWords describes a list of alternative spellings for a
// canonical (all lowercase) spelling of a word.
type AltWords struct {
	Canon	string;		// canonical word spelling (all lowercase)
	Alts	[]string;	// alternative spelling for the same word
}


func lessWordPair(x, y interface{}) bool	{ return x.(*wordPair).canon < y.(*wordPair).canon }


// newAltWords allocates a new *AltWords from the *wordPair run [i, j) in h.
func newAltWords(h *RunList, i, j int) interface{} {
	canon := h.At(i).(*wordPair).canon;
	alts := make([]string, j-i);
	k := 0;
	for ; i < j; i++ {
		alts[k] = h.At(i).(*wordPair).alt;
		k++;
	}
	return &AltWords{canon, alts};
}


func (a *AltWords) filter(s string) *AltWords {
	if len(a.Alts) == 1 && a.Alts[0] == s {
		// there are no different alternatives
		return nil;
	}

	// make a new AltWords with the current spelling removed
	alts := make([]string, len(a.Alts));
	i := 0;
	for _, w := range a.Alts {
		if w != s {
			alts[i] = w;
			i++;
		}
	}
	return &AltWords{a.Canon, alts[0:i]};
}


// ----------------------------------------------------------------------------
// Indexer

type IndexResult struct {
	Decls	RunList;	// package-level declarations (with snippets)
	Others	RunList;	// all other occurences
}


// An Indexer maintains the data structures and provides the machinery
// for indexing .go files under a file tree. It implements the path.Visitor
// interface for walking file trees, and the ast.Visitor interface for
// walking Go ASTs.
type Indexer struct {
	words		map[string]*IndexResult;	// RunLists of Spots
	snippets	vector.Vector;			// vector of *Snippets, indexed by snippet indices
	file		*File;				// current file
	decl		ast.Decl;			// current decl
	nspots		int;				// number of spots encountered
}


func (x *Indexer) addSnippet(s *Snippet) int {
	index := x.snippets.Len();
	x.snippets.Push(s);
	return index;
}


func (x *Indexer) visitComment(c *ast.CommentGroup) {
	if c != nil {
		ast.Walk(x, c);
	}
}


func (x *Indexer) visitIdent(kind SpotKind, id *ast.Ident) {
	if id != nil {
		lists, found := x.words[id.Value];
		if !found {
			lists = new(IndexResult);
			x.words[id.Value] = lists;
		}

		if kind == Use || x.decl == nil {
			// not a declaration or no snippet required
			info := makeSpotInfo(kind, id.Pos().Line, false);
			lists.Others.Push(Spot{x.file, info});
		} else {
			// a declaration with snippet
			index := x.addSnippet(NewSnippet(x.decl, id));
			info := makeSpotInfo(kind, index, true);
			lists.Decls.Push(Spot{x.file, info});
		}

		x.nspots++;
	}
}


func (x *Indexer) visitSpec(spec ast.Spec, isVarDecl bool) {
	switch n := spec.(type) {
	case *ast.ImportSpec:
		x.visitComment(n.Doc);
		x.visitIdent(ImportDecl, n.Name);
		for _, s := range n.Path {
			ast.Walk(x, s);
		}
		x.visitComment(n.Comment);

	case *ast.ValueSpec:
		x.visitComment(n.Doc);
		kind := ConstDecl;
		if isVarDecl {
			kind = VarDecl;
		}
		for _, n := range n.Names {
			x.visitIdent(kind, n);
		}
		ast.Walk(x, n.Type);
		for _, v := range n.Values {
			ast.Walk(x, v);
		}
		x.visitComment(n.Comment);

	case *ast.TypeSpec:
		x.visitComment(n.Doc);
		x.visitIdent(TypeDecl, n.Name);
		ast.Walk(x, n.Type);
		x.visitComment(n.Comment);
	}
}


func (x *Indexer) Visit(node interface{}) bool {
	// TODO(gri): methods in interface types are categorized as VarDecl
	switch n := node.(type) {
	case *ast.Ident:
		x.visitIdent(Use, n);

	case *ast.Field:
		x.decl = nil;	// no snippets for fields
		x.visitComment(n.Doc);
		for _, m := range n.Names {
			x.visitIdent(VarDecl, m);
		}
		ast.Walk(x, n.Type);
		for _, s := range n.Tag {
			ast.Walk(x, s);
		}
		x.visitComment(n.Comment);

	case *ast.DeclStmt:
		if decl, ok := n.Decl.(*ast.GenDecl); ok {
			// local declarations can only be *ast.GenDecls
			x.decl = nil;	// no snippets for local declarations
			x.visitComment(decl.Doc);
			for _, s := range decl.Specs {
				x.visitSpec(s, decl.Tok == token.VAR);
			}
		} else {
			// handle error case gracefully
			ast.Walk(x, n.Decl);
		}

	case *ast.GenDecl:
		x.decl = n;
		x.visitComment(n.Doc);
		for _, s := range n.Specs {
			x.visitSpec(s, n.Tok == token.VAR);
		}

	case *ast.FuncDecl:
		x.visitComment(n.Doc);
		kind := FuncDecl;
		if n.Recv != nil {
			kind = MethodDecl;
			ast.Walk(x, n.Recv);
		}
		x.decl = n;
		x.visitIdent(kind, n.Name);
		ast.Walk(x, n.Type);
		if n.Body != nil {
			ast.Walk(x, n.Type);
		}

	case *ast.File:
		x.visitComment(n.Doc);
		x.decl = nil;
		x.visitIdent(PackageClause, n.Name);
		for _, d := range n.Decls {
			ast.Walk(x, d);
		}
		// don't visit package level comments for now
		// to avoid duplicate visiting from individual
		// nodes

	default:
		return true;
	}

	return false;
}


func (x *Indexer) VisitDir(path string, d *os.Dir) bool {
	return true;
}


func (x *Indexer) VisitFile(path string, d *os.Dir) {
	if !isGoFile(d) {
		return;
	}

	file, err := parser.ParseFile(path, nil, parser.ParseComments);
	if err != nil {
		return;	// ignore files with (parse) errors
	}

	dir, _ := pathutil.Split(path);
	pak := Pak{dir, file.Name.Value};
	x.file = &File{path, pak};
	ast.Walk(x, file);
}


// ----------------------------------------------------------------------------
// Index

type LookupResult struct {
	Decls	HitList;	// package-level declarations (with snippets)
	Others	HitList;	// all other occurences
}


type Index struct {
	words		map[string]*LookupResult;	// maps words to hit lists
	alts		map[string]*AltWords;		// maps canonical(words) to lists of alternative spellings
	snippets	[]*Snippet;			// all snippets, indexed by snippet index
	nspots		int;				// number of spots indexed (a measure of the index size)
}


func canonical(w string) string	{ return strings.ToLower(w) }


// NewIndex creates a new index for the file tree rooted at root.
func NewIndex(root string) *Index {
	var x Indexer;

	// initialize Indexer
	x.words = make(map[string]*IndexResult);

	// collect all Spots
	pathutil.Walk(root, &x, nil);

	// for each word, reduce the RunLists into a LookupResult;
	// also collect the word with its canonical spelling in a
	// word list for later computation of alternative spellings
	words := make(map[string]*LookupResult);
	var wlist RunList;
	for w, h := range x.words {
		decls := reduce(&h.Decls);
		others := reduce(&h.Others);
		words[w] = &LookupResult{
			Decls: decls,
			Others: others,
		};
		wlist.Push(&wordPair{canonical(w), w});
	}

	// reduce the word list {canonical(w), w} into
	// a list of AltWords runs {canonical(w), {w}}
	alist := wlist.reduce(lessWordPair, newAltWords);

	// convert alist into a map of alternative spellings
	alts := make(map[string]*AltWords);
	for i := 0; i < alist.Len(); i++ {
		a := alist.At(i).(*AltWords);
		alts[a.Canon] = a;
	}

	// convert snippet vector into a list
	snippets := make([]*Snippet, x.snippets.Len());
	for i := 0; i < x.snippets.Len(); i++ {
		snippets[i] = x.snippets.At(i).(*Snippet);
	}

	return &Index{words, alts, snippets, x.nspots};
}


// Size returns the number of different words and
// spots indexed as a measure for the index size.
func (x *Index) Size() (nwords int, nspots int) {
	return len(x.words), x.nspots;
}


func (x *Index) LookupWord(w string) (match *LookupResult, alt *AltWords) {
	match, _ = x.words[w];
	alt, _ = x.alts[canonical(w)];
	// remove current spelling from alternatives
	// (if there is no match, the alternatives do
	// not contain the current spelling)
	if match != nil && alt != nil {
		alt = alt.filter(w);
	}
	return;
}


// For a given string s, which is either a single identifier or a qualified
// identifier, Lookup returns a LookupResult, and a list of alternative
// spellings, if any.
func (x *Index) Lookup(s string) (match *LookupResult, alt *AltWords) {
	ss := strings.Split(s, ".", 0);

	switch len(ss) {
	case 1:
		match, alt = x.LookupWord(ss[0]);

	case 2:
		pakname := ss[0];
		match, alt = x.LookupWord(ss[1]);
		if match != nil {
			// found a match - filter by package name
			decls := match.Decls.filter(pakname);
			others := match.Others.filter(pakname);
			match = &LookupResult{decls, others};
		}
		if alt != nil {
			// alternative spellings found - add package name
			// TODO(gri): At the moment this is not very smart
			// and likely will produce suggestions that have
			// no match. Should filter incorrect alternatives.
			canon := pakname + "." + alt.Canon;	// for completeness (currently not used)
			alts := make([]string, len(alt.Alts));
			for i, a := range alt.Alts {
				alts[i] = pakname+"."+a;
			}
			alt = &AltWords{canon, alts};
		}
	}

	return;
}


func (x *Index) Snippet(i int) *Snippet {
	// handle illegal snippet indices gracefully
	if 0 <= i && i < len(x.snippets) {
		return x.snippets[i];
	}
	return nil;
}
