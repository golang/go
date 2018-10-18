// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package godoc

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"go/ast"
	"go/build"
	"go/doc"
	"go/token"
	htmlpkg "html"
	htmltemplate "html/template"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	pathpkg "path"
	"path/filepath"
	"sort"
	"strings"
	"text/template"
	"time"

	"golang.org/x/tools/godoc/analysis"
	"golang.org/x/tools/godoc/util"
	"golang.org/x/tools/godoc/vfs"
)

// handlerServer is a migration from an old godoc http Handler type.
// This should probably merge into something else.
type handlerServer struct {
	p           *Presentation
	c           *Corpus  // copy of p.Corpus
	pattern     string   // url pattern; e.g. "/pkg/"
	stripPrefix string   // prefix to strip from import path; e.g. "pkg/"
	fsRoot      string   // file system root to which the pattern is mapped; e.g. "/src"
	exclude     []string // file system paths to exclude; e.g. "/src/cmd"
}

func (s *handlerServer) registerWithMux(mux *http.ServeMux) {
	mux.Handle(s.pattern, s)
}

// getPageInfo returns the PageInfo for a package directory abspath. If the
// parameter genAST is set, an AST containing only the package exports is
// computed (PageInfo.PAst), otherwise package documentation (PageInfo.Doc)
// is extracted from the AST. If there is no corresponding package in the
// directory, PageInfo.PAst and PageInfo.PDoc are nil. If there are no sub-
// directories, PageInfo.Dirs is nil. If an error occurred, PageInfo.Err is
// set to the respective error but the error is not logged.
//
func (h *handlerServer) GetPageInfo(abspath, relpath string, mode PageInfoMode, goos, goarch string) *PageInfo {
	info := &PageInfo{Dirname: abspath, Mode: mode}

	// Restrict to the package files that would be used when building
	// the package on this system.  This makes sure that if there are
	// separate implementations for, say, Windows vs Unix, we don't
	// jumble them all together.
	// Note: If goos/goarch aren't set, the current binary's GOOS/GOARCH
	// are used.
	ctxt := build.Default
	ctxt.IsAbsPath = pathpkg.IsAbs
	ctxt.IsDir = func(path string) bool {
		fi, err := h.c.fs.Stat(filepath.ToSlash(path))
		return err == nil && fi.IsDir()
	}
	ctxt.ReadDir = func(dir string) ([]os.FileInfo, error) {
		f, err := h.c.fs.ReadDir(filepath.ToSlash(dir))
		filtered := make([]os.FileInfo, 0, len(f))
		for _, i := range f {
			if mode&NoFiltering != 0 || i.Name() != "internal" {
				filtered = append(filtered, i)
			}
		}
		return filtered, err
	}
	ctxt.OpenFile = func(name string) (r io.ReadCloser, err error) {
		data, err := vfs.ReadFile(h.c.fs, filepath.ToSlash(name))
		if err != nil {
			return nil, err
		}
		return ioutil.NopCloser(bytes.NewReader(data)), nil
	}

	// Make the syscall/js package always visible by default.
	// It defaults to the host's GOOS/GOARCH, and golang.org's
	// linux/amd64 means the wasm syscall/js package was blank.
	// And you can't run godoc on js/wasm anyway, so host defaults
	// don't make sense here.
	if goos == "" && goarch == "" && relpath == "syscall/js" {
		goos, goarch = "js", "wasm"
	}
	if goos != "" {
		ctxt.GOOS = goos
	}
	if goarch != "" {
		ctxt.GOARCH = goarch
	}

	pkginfo, err := ctxt.ImportDir(abspath, 0)
	// continue if there are no Go source files; we still want the directory info
	if _, nogo := err.(*build.NoGoError); err != nil && !nogo {
		info.Err = err
		return info
	}

	// collect package files
	pkgname := pkginfo.Name
	pkgfiles := append(pkginfo.GoFiles, pkginfo.CgoFiles...)
	if len(pkgfiles) == 0 {
		// Commands written in C have no .go files in the build.
		// Instead, documentation may be found in an ignored file.
		// The file may be ignored via an explicit +build ignore
		// constraint (recommended), or by defining the package
		// documentation (historic).
		pkgname = "main" // assume package main since pkginfo.Name == ""
		pkgfiles = pkginfo.IgnoredGoFiles
	}

	// get package information, if any
	if len(pkgfiles) > 0 {
		// build package AST
		fset := token.NewFileSet()
		files, err := h.c.parseFiles(fset, relpath, abspath, pkgfiles)
		if err != nil {
			info.Err = err
			return info
		}

		// ignore any errors - they are due to unresolved identifiers
		pkg, _ := ast.NewPackage(fset, files, poorMansImporter, nil)

		// extract package documentation
		info.FSet = fset
		if mode&ShowSource == 0 {
			// show extracted documentation
			var m doc.Mode
			if mode&NoFiltering != 0 {
				m |= doc.AllDecls
			}
			if mode&AllMethods != 0 {
				m |= doc.AllMethods
			}
			info.PDoc = doc.New(pkg, pathpkg.Clean(relpath), m) // no trailing '/' in importpath
			if mode&NoTypeAssoc != 0 {
				for _, t := range info.PDoc.Types {
					info.PDoc.Consts = append(info.PDoc.Consts, t.Consts...)
					info.PDoc.Vars = append(info.PDoc.Vars, t.Vars...)
					info.PDoc.Funcs = append(info.PDoc.Funcs, t.Funcs...)
					t.Consts = nil
					t.Vars = nil
					t.Funcs = nil
				}
				// for now we cannot easily sort consts and vars since
				// go/doc.Value doesn't export the order information
				sort.Sort(funcsByName(info.PDoc.Funcs))
			}

			// collect examples
			testfiles := append(pkginfo.TestGoFiles, pkginfo.XTestGoFiles...)
			files, err = h.c.parseFiles(fset, relpath, abspath, testfiles)
			if err != nil {
				log.Println("parsing examples:", err)
			}
			info.Examples = collectExamples(h.c, pkg, files)

			// collect any notes that we want to show
			if info.PDoc.Notes != nil {
				// could regexp.Compile only once per godoc, but probably not worth it
				if rx := h.p.NotesRx; rx != nil {
					for m, n := range info.PDoc.Notes {
						if rx.MatchString(m) {
							if info.Notes == nil {
								info.Notes = make(map[string][]*doc.Note)
							}
							info.Notes[m] = n
						}
					}
				}
			}

		} else {
			// show source code
			// TODO(gri) Consider eliminating export filtering in this mode,
			//           or perhaps eliminating the mode altogether.
			if mode&NoFiltering == 0 {
				packageExports(fset, pkg)
			}
			info.PAst = files
		}
		info.IsMain = pkgname == "main"
	}

	// get directory information, if any
	var dir *Directory
	var timestamp time.Time
	if tree, ts := h.c.fsTree.Get(); tree != nil && tree.(*Directory) != nil {
		// directory tree is present; lookup respective directory
		// (may still fail if the file system was updated and the
		// new directory tree has not yet been computed)
		dir = tree.(*Directory).lookup(abspath)
		timestamp = ts
	}
	if dir == nil {
		// TODO(agnivade): handle this case better, now since there is no CLI mode.
		// no directory tree present (happens in command-line mode);
		// compute 2 levels for this page. The second level is to
		// get the synopses of sub-directories.
		// note: cannot use path filter here because in general
		// it doesn't contain the FSTree path
		dir = h.c.newDirectory(abspath, 2)
		timestamp = time.Now()
	}
	info.Dirs = dir.listing(true, func(path string) bool { return h.includePath(path, mode) })

	info.DirTime = timestamp
	info.DirFlat = mode&FlatDir != 0

	return info
}

func (h *handlerServer) includePath(path string, mode PageInfoMode) (r bool) {
	// if the path is under one of the exclusion paths, don't list.
	for _, e := range h.exclude {
		if strings.HasPrefix(path, e) {
			return false
		}
	}

	// if the path includes 'internal', don't list unless we are in the NoFiltering mode.
	if mode&NoFiltering != 0 {
		return true
	}
	if strings.Contains(path, "internal") || strings.Contains(path, "vendor") {
		for _, c := range strings.Split(filepath.Clean(path), string(os.PathSeparator)) {
			if c == "internal" || c == "vendor" {
				return false
			}
		}
	}
	return true
}

type funcsByName []*doc.Func

func (s funcsByName) Len() int           { return len(s) }
func (s funcsByName) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }
func (s funcsByName) Less(i, j int) bool { return s[i].Name < s[j].Name }

func (h *handlerServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if redirect(w, r) {
		return
	}

	relpath := pathpkg.Clean(r.URL.Path[len(h.stripPrefix)+1:])

	if !h.corpusInitialized() {
		h.p.ServeError(w, r, relpath, errors.New("Scan is not yet complete. Please retry after a few moments"))
		return
	}

	abspath := pathpkg.Join(h.fsRoot, relpath)
	mode := h.p.GetPageInfoMode(r)
	if relpath == builtinPkgPath {
		mode = NoFiltering | NoTypeAssoc
	}
	info := h.GetPageInfo(abspath, relpath, mode, r.FormValue("GOOS"), r.FormValue("GOARCH"))
	if info.Err != nil {
		log.Print(info.Err)
		h.p.ServeError(w, r, relpath, info.Err)
		return
	}

	var tabtitle, title, subtitle string
	switch {
	case info.PAst != nil:
		for _, ast := range info.PAst {
			tabtitle = ast.Name.Name
			break
		}
	case info.PDoc != nil:
		tabtitle = info.PDoc.Name
	default:
		tabtitle = info.Dirname
		title = "Directory "
		if h.p.ShowTimestamps {
			subtitle = "Last update: " + info.DirTime.String()
		}
	}
	if title == "" {
		if info.IsMain {
			// assume that the directory name is the command name
			_, tabtitle = pathpkg.Split(relpath)
			title = "Command "
		} else {
			title = "Package "
		}
	}
	title += tabtitle

	// special cases for top-level package/command directories
	switch tabtitle {
	case "/src":
		title = "Packages"
		tabtitle = "Packages"
	case "/src/cmd":
		title = "Commands"
		tabtitle = "Commands"
	}

	// Emit JSON array for type information.
	pi := h.c.Analysis.PackageInfo(relpath)
	hasTreeView := len(pi.CallGraph) != 0
	info.CallGraphIndex = pi.CallGraphIndex
	info.CallGraph = htmltemplate.JS(marshalJSON(pi.CallGraph))
	info.AnalysisData = htmltemplate.JS(marshalJSON(pi.Types))
	info.TypeInfoIndex = make(map[string]int)
	for i, ti := range pi.Types {
		info.TypeInfoIndex[ti.Name] = i
	}

	info.GoogleCN = googleCN(r)
	var body []byte
	if info.Dirname == "/src" {
		body = applyTemplate(h.p.PackageRootHTML, "packageRootHTML", info)
	} else {
		body = applyTemplate(h.p.PackageHTML, "packageHTML", info)
	}
	h.p.ServePage(w, Page{
		Title:    title,
		Tabtitle: tabtitle,
		Subtitle: subtitle,
		Body:     body,
		GoogleCN: info.GoogleCN,
		TreeView: hasTreeView,
	})
}

func (h *handlerServer) corpusInitialized() bool {
	h.c.initMu.RLock()
	defer h.c.initMu.RUnlock()
	return h.c.initDone
}

type PageInfoMode uint

const (
	PageInfoModeQueryString = "m" // query string where PageInfoMode is stored

	NoFiltering PageInfoMode = 1 << iota // do not filter exports
	AllMethods                           // show all embedded methods
	ShowSource                           // show source code, do not extract documentation
	NoHTML                               // show result in textual form, do not generate HTML
	FlatDir                              // show directory in a flat (non-indented) manner
	NoTypeAssoc                          // don't associate consts, vars, and factory functions with types
)

// modeNames defines names for each PageInfoMode flag.
var modeNames = map[string]PageInfoMode{
	"all":     NoFiltering,
	"methods": AllMethods,
	"src":     ShowSource,
	"text":    NoHTML,
	"flat":    FlatDir,
}

// generate a query string for persisting PageInfoMode between pages.
func modeQueryString(mode PageInfoMode) string {
	if modeNames := mode.names(); len(modeNames) > 0 {
		return "?m=" + strings.Join(modeNames, ",")
	}
	return ""
}

// alphabetically sorted names of active flags for a PageInfoMode.
func (m PageInfoMode) names() []string {
	var names []string
	for name, mode := range modeNames {
		if m&mode != 0 {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	return names
}

// GetPageInfoMode computes the PageInfoMode flags by analyzing the request
// URL form value "m". It is value is a comma-separated list of mode names
// as defined by modeNames (e.g.: m=src,text).
func (p *Presentation) GetPageInfoMode(r *http.Request) PageInfoMode {
	var mode PageInfoMode
	for _, k := range strings.Split(r.FormValue(PageInfoModeQueryString), ",") {
		if m, found := modeNames[strings.TrimSpace(k)]; found {
			mode |= m
		}
	}
	if p.AdjustPageInfoMode != nil {
		mode = p.AdjustPageInfoMode(r, mode)
	}
	return mode
}

// poorMansImporter returns a (dummy) package object named
// by the last path component of the provided package path
// (as is the convention for packages). This is sufficient
// to resolve package identifiers without doing an actual
// import. It never returns an error.
//
func poorMansImporter(imports map[string]*ast.Object, path string) (*ast.Object, error) {
	pkg := imports[path]
	if pkg == nil {
		// note that strings.LastIndex returns -1 if there is no "/"
		pkg = ast.NewObj(ast.Pkg, path[strings.LastIndex(path, "/")+1:])
		pkg.Data = ast.NewScope(nil) // required by ast.NewPackage for dot-import
		imports[path] = pkg
	}
	return pkg, nil
}

// globalNames returns a set of the names declared by all package-level
// declarations. Method names are returned in the form Receiver_Method.
func globalNames(pkg *ast.Package) map[string]bool {
	names := make(map[string]bool)
	for _, file := range pkg.Files {
		for _, decl := range file.Decls {
			addNames(names, decl)
		}
	}
	return names
}

// collectExamples collects examples for pkg from testfiles.
func collectExamples(c *Corpus, pkg *ast.Package, testfiles map[string]*ast.File) []*doc.Example {
	var files []*ast.File
	for _, f := range testfiles {
		files = append(files, f)
	}

	var examples []*doc.Example
	globals := globalNames(pkg)
	for _, e := range doc.Examples(files...) {
		name := stripExampleSuffix(e.Name)
		if name == "" || globals[name] {
			examples = append(examples, e)
		} else if c.Verbose {
			log.Printf("skipping example 'Example%s' because '%s' is not a known function or type", e.Name, e.Name)
		}
	}

	return examples
}

// addNames adds the names declared by decl to the names set.
// Method names are added in the form ReceiverTypeName_Method.
func addNames(names map[string]bool, decl ast.Decl) {
	switch d := decl.(type) {
	case *ast.FuncDecl:
		name := d.Name.Name
		if d.Recv != nil {
			var typeName string
			switch r := d.Recv.List[0].Type.(type) {
			case *ast.StarExpr:
				typeName = r.X.(*ast.Ident).Name
			case *ast.Ident:
				typeName = r.Name
			}
			name = typeName + "_" + name
		}
		names[name] = true
	case *ast.GenDecl:
		for _, spec := range d.Specs {
			switch s := spec.(type) {
			case *ast.TypeSpec:
				names[s.Name.Name] = true
			case *ast.ValueSpec:
				for _, id := range s.Names {
					names[id.Name] = true
				}
			}
		}
	}
}

// packageExports is a local implementation of ast.PackageExports
// which correctly updates each package file's comment list.
// (The ast.PackageExports signature is frozen, hence the local
// implementation).
//
func packageExports(fset *token.FileSet, pkg *ast.Package) {
	for _, src := range pkg.Files {
		cmap := ast.NewCommentMap(fset, src, src.Comments)
		ast.FileExports(src)
		src.Comments = cmap.Filter(src).Comments()
	}
}

func applyTemplate(t *template.Template, name string, data interface{}) []byte {
	var buf bytes.Buffer
	if err := t.Execute(&buf, data); err != nil {
		log.Printf("%s.Execute: %s", name, err)
	}
	return buf.Bytes()
}

type writerCapturesErr struct {
	w   io.Writer
	err error
}

func (w *writerCapturesErr) Write(p []byte) (int, error) {
	n, err := w.w.Write(p)
	if err != nil {
		w.err = err
	}
	return n, err
}

// applyTemplateToResponseWriter uses an http.ResponseWriter as the io.Writer
// for the call to template.Execute.  It uses an io.Writer wrapper to capture
// errors from the underlying http.ResponseWriter.  Errors are logged only when
// they come from the template processing and not the Writer; this avoid
// polluting log files with error messages due to networking issues, such as
// client disconnects and http HEAD protocol violations.
func applyTemplateToResponseWriter(rw http.ResponseWriter, t *template.Template, data interface{}) {
	w := &writerCapturesErr{w: rw}
	err := t.Execute(w, data)
	// There are some cases where template.Execute does not return an error when
	// rw returns an error, and some where it does.  So check w.err first.
	if w.err == nil && err != nil {
		// Log template errors.
		log.Printf("%s.Execute: %s", t.Name(), err)
	}
}

func redirect(w http.ResponseWriter, r *http.Request) (redirected bool) {
	canonical := pathpkg.Clean(r.URL.Path)
	if !strings.HasSuffix(canonical, "/") {
		canonical += "/"
	}
	if r.URL.Path != canonical {
		url := *r.URL
		url.Path = canonical
		http.Redirect(w, r, url.String(), http.StatusMovedPermanently)
		redirected = true
	}
	return
}

func redirectFile(w http.ResponseWriter, r *http.Request) (redirected bool) {
	c := pathpkg.Clean(r.URL.Path)
	c = strings.TrimRight(c, "/")
	if r.URL.Path != c {
		url := *r.URL
		url.Path = c
		http.Redirect(w, r, url.String(), http.StatusMovedPermanently)
		redirected = true
	}
	return
}

func (p *Presentation) serveTextFile(w http.ResponseWriter, r *http.Request, abspath, relpath, title string) {
	src, err := vfs.ReadFile(p.Corpus.fs, abspath)
	if err != nil {
		log.Printf("ReadFile: %s", err)
		p.ServeError(w, r, relpath, err)
		return
	}

	if r.FormValue(PageInfoModeQueryString) == "text" {
		p.ServeText(w, src)
		return
	}

	h := r.FormValue("h")
	s := RangeSelection(r.FormValue("s"))

	var buf bytes.Buffer
	if pathpkg.Ext(abspath) == ".go" {
		// Find markup links for this file (e.g. "/src/fmt/print.go").
		fi := p.Corpus.Analysis.FileInfo(abspath)
		buf.WriteString("<script type='text/javascript'>document.ANALYSIS_DATA = ")
		buf.Write(marshalJSON(fi.Data))
		buf.WriteString(";</script>\n")

		if status := p.Corpus.Analysis.Status(); status != "" {
			buf.WriteString("<a href='/lib/godoc/analysis/help.html'>Static analysis features</a> ")
			// TODO(adonovan): show analysis status at per-file granularity.
			fmt.Fprintf(&buf, "<span style='color: grey'>[%s]</span><br/>", htmlpkg.EscapeString(status))
		}

		buf.WriteString("<pre>")
		formatGoSource(&buf, src, fi.Links, h, s)
		buf.WriteString("</pre>")
	} else {
		buf.WriteString("<pre>")
		FormatText(&buf, src, 1, false, h, s)
		buf.WriteString("</pre>")
	}
	fmt.Fprintf(&buf, `<p><a href="/%s?m=text">View as plain text</a></p>`, htmlpkg.EscapeString(relpath))

	p.ServePage(w, Page{
		Title:    title,
		SrcPath:  relpath,
		Tabtitle: relpath,
		Body:     buf.Bytes(),
		GoogleCN: googleCN(r),
	})
}

// formatGoSource HTML-escapes Go source text and writes it to w,
// decorating it with the specified analysis links.
//
func formatGoSource(buf *bytes.Buffer, text []byte, links []analysis.Link, pattern string, selection Selection) {
	// Emit to a temp buffer so that we can add line anchors at the end.
	saved, buf := buf, new(bytes.Buffer)

	var i int
	var link analysis.Link // shared state of the two funcs below
	segmentIter := func() (seg Segment) {
		if i < len(links) {
			link = links[i]
			i++
			seg = Segment{link.Start(), link.End()}
		}
		return
	}
	linkWriter := func(w io.Writer, offs int, start bool) {
		link.Write(w, offs, start)
	}

	comments := tokenSelection(text, token.COMMENT)
	var highlights Selection
	if pattern != "" {
		highlights = regexpSelection(text, pattern)
	}

	FormatSelections(buf, text, linkWriter, segmentIter, selectionTag, comments, highlights, selection)

	// Now copy buf to saved, adding line anchors.

	// The lineSelection mechanism can't be composed with our
	// linkWriter, so we have to add line spans as another pass.
	n := 1
	for _, line := range bytes.Split(buf.Bytes(), []byte("\n")) {
		// The line numbers are inserted into the document via a CSS ::before
		// pseudo-element. This prevents them from being copied when users
		// highlight and copy text.
		// ::before is supported in 98% of browsers: https://caniuse.com/#feat=css-gencontent
		// This is also the trick Github uses to hide line numbers.
		//
		// The first tab for the code snippet needs to start in column 9, so
		// it indents a full 8 spaces, hence the two nbsp's. Otherwise the tab
		// character only indents a short amount.
		//
		// Due to rounding and font width Firefox might not treat 8 rendered
		// characters as 8 characters wide, and subsequently may treat the tab
		// character in the 9th position as moving the width from (7.5 or so) up
		// to 8. See
		// https://github.com/webcompat/web-bugs/issues/17530#issuecomment-402675091
		// for a fuller explanation. The solution is to add a CSS class to
		// explicitly declare the width to be 8 characters.
		fmt.Fprintf(saved, `<span id="L%d" class="ln">%6d&nbsp;&nbsp;</span>`, n, n)
		n++
		saved.Write(line)
		saved.WriteByte('\n')
	}
}

func (p *Presentation) serveDirectory(w http.ResponseWriter, r *http.Request, abspath, relpath string) {
	if redirect(w, r) {
		return
	}

	list, err := p.Corpus.fs.ReadDir(abspath)
	if err != nil {
		p.ServeError(w, r, relpath, err)
		return
	}

	p.ServePage(w, Page{
		Title:    "Directory",
		SrcPath:  relpath,
		Tabtitle: relpath,
		Body:     applyTemplate(p.DirlistHTML, "dirlistHTML", list),
		GoogleCN: googleCN(r),
	})
}

func (p *Presentation) ServeHTMLDoc(w http.ResponseWriter, r *http.Request, abspath, relpath string) {
	// get HTML body contents
	src, err := vfs.ReadFile(p.Corpus.fs, abspath)
	if err != nil {
		log.Printf("ReadFile: %s", err)
		p.ServeError(w, r, relpath, err)
		return
	}

	// if it begins with "<!DOCTYPE " assume it is standalone
	// html that doesn't need the template wrapping.
	if bytes.HasPrefix(src, doctype) {
		w.Write(src)
		return
	}

	// if it begins with a JSON blob, read in the metadata.
	meta, src, err := extractMetadata(src)
	if err != nil {
		log.Printf("decoding metadata %s: %v", relpath, err)
	}

	page := Page{
		Title:    meta.Title,
		Subtitle: meta.Subtitle,
		GoogleCN: googleCN(r),
	}

	// evaluate as template if indicated
	if meta.Template {
		tmpl, err := template.New("main").Funcs(p.TemplateFuncs()).Parse(string(src))
		if err != nil {
			log.Printf("parsing template %s: %v", relpath, err)
			p.ServeError(w, r, relpath, err)
			return
		}
		var buf bytes.Buffer
		if err := tmpl.Execute(&buf, page); err != nil {
			log.Printf("executing template %s: %v", relpath, err)
			p.ServeError(w, r, relpath, err)
			return
		}
		src = buf.Bytes()
	}

	// if it's the language spec, add tags to EBNF productions
	if strings.HasSuffix(abspath, "go_spec.html") {
		var buf bytes.Buffer
		Linkify(&buf, src)
		src = buf.Bytes()
	}

	page.Body = src
	p.ServePage(w, page)
}

func (p *Presentation) ServeFile(w http.ResponseWriter, r *http.Request) {
	p.serveFile(w, r)
}

func (p *Presentation) serveFile(w http.ResponseWriter, r *http.Request) {
	relpath := r.URL.Path

	// Check to see if we need to redirect or serve another file.
	if m := p.Corpus.MetadataFor(relpath); m != nil {
		if m.Path != relpath {
			// Redirect to canonical path.
			http.Redirect(w, r, m.Path, http.StatusMovedPermanently)
			return
		}
		// Serve from the actual filesystem path.
		relpath = m.filePath
	}

	abspath := relpath
	relpath = relpath[1:] // strip leading slash

	switch pathpkg.Ext(relpath) {
	case ".html":
		if strings.HasSuffix(relpath, "/index.html") {
			// We'll show index.html for the directory.
			// Use the dir/ version as canonical instead of dir/index.html.
			http.Redirect(w, r, r.URL.Path[0:len(r.URL.Path)-len("index.html")], http.StatusMovedPermanently)
			return
		}
		p.ServeHTMLDoc(w, r, abspath, relpath)
		return

	case ".go":
		p.serveTextFile(w, r, abspath, relpath, "Source file")
		return
	}

	dir, err := p.Corpus.fs.Lstat(abspath)
	if err != nil {
		log.Print(err)
		p.ServeError(w, r, relpath, err)
		return
	}

	if dir != nil && dir.IsDir() {
		if redirect(w, r) {
			return
		}
		if index := pathpkg.Join(abspath, "index.html"); util.IsTextFile(p.Corpus.fs, index) {
			p.ServeHTMLDoc(w, r, index, index)
			return
		}
		p.serveDirectory(w, r, abspath, relpath)
		return
	}

	if util.IsTextFile(p.Corpus.fs, abspath) {
		if redirectFile(w, r) {
			return
		}
		p.serveTextFile(w, r, abspath, relpath, "Text file")
		return
	}

	p.fileServer.ServeHTTP(w, r)
}

func (p *Presentation) ServeText(w http.ResponseWriter, text []byte) {
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.Write(text)
}

func marshalJSON(x interface{}) []byte {
	var data []byte
	var err error
	const indentJSON = false // for easier debugging
	if indentJSON {
		data, err = json.MarshalIndent(x, "", "    ")
	} else {
		data, err = json.Marshal(x)
	}
	if err != nil {
		panic(fmt.Sprintf("json.Marshal failed: %s", err))
	}
	return data
}
