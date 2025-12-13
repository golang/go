// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"errors"
	"fmt"
	"internal/buildcfg"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/objabi"
)

func LoadPackage(filenames []string) {
	base.Timer.Start("fe", "parse")

	// Limit the number of simultaneously open files.
	sem := make(chan struct{}, runtime.GOMAXPROCS(0)+10)

	noders := make([]*noder, len(filenames))
	for i := range noders {
		p := noder{
			err: make(chan syntax.Error),
		}
		noders[i] = &p
	}

	// Move the entire syntax processing logic into a separate goroutine to avoid blocking on the "sem".
	go func() {
		for i, filename := range filenames {
			filename := filename
			p := noders[i]
			sem <- struct{}{}
			go func() {
				defer func() { <-sem }()
				defer close(p.err)
				fbase := syntax.NewFileBase(filename)

				f, err := os.Open(filename)
				if err != nil {
					p.error(syntax.Error{Msg: err.Error()})
					return
				}
				defer f.Close()

				p.file, _ = syntax.Parse(fbase, f, p.error, p.pragma, syntax.CheckBranches) // errors are tracked via p.error
				// Rewrite question expressions (?.  and ? :) to standard Go syntax
				if p.file != nil {
					syntax.RewriteQuestionExprs(p.file)
				}
			}()
		}
	}()

	var lines uint
	var m posMap
	for _, p := range noders {
		for e := range p.err {
			base.ErrorfAt(m.makeXPos(e.Pos), 0, "%s", e.Msg)
		}
		if p.file == nil {
			base.ErrorExit()
		}
		lines += p.file.EOF.Line()
	}
	base.Timer.AddEvent(int64(lines), "lines")

	unified(m, noders)
}

// trimFilename returns the "trimmed" filename of b, which is the
// absolute filename after applying -trimpath processing. This
// filename form is suitable for use in object files and export data.
//
// If b's filename has already been trimmed (i.e., because it was read
// in from an imported package's export data), then the filename is
// returned unchanged.
func trimFilename(b *syntax.PosBase) string {
	filename := b.Filename()
	if !b.Trimmed() {
		dir := ""
		if b.IsFileBase() {
			dir = base.Ctxt.Pathname
		}
		filename = objabi.AbsFile(dir, filename, base.Flag.TrimPath)
	}
	return filename
}

// noder transforms package syntax's AST into a Node tree.
type noder struct {
	file       *syntax.File
	linknames  []linkname
	pragcgobuf [][]string
	err        chan syntax.Error
}

// linkname records a //go:linkname directive.
type linkname struct {
	pos    syntax.Pos
	local  string
	remote string
}

var unOps = [...]ir.Op{
	syntax.Recv: ir.ORECV,
	syntax.Mul:  ir.ODEREF,
	syntax.And:  ir.OADDR,

	syntax.Not: ir.ONOT,
	syntax.Xor: ir.OBITNOT,
	syntax.Add: ir.OPLUS,
	syntax.Sub: ir.ONEG,
}

var binOps = [...]ir.Op{
	syntax.OrOr:   ir.OOROR,
	syntax.AndAnd: ir.OANDAND,

	syntax.Eql: ir.OEQ,
	syntax.Neq: ir.ONE,
	syntax.Lss: ir.OLT,
	syntax.Leq: ir.OLE,
	syntax.Gtr: ir.OGT,
	syntax.Geq: ir.OGE,

	syntax.Add: ir.OADD,
	syntax.Sub: ir.OSUB,
	syntax.Or:  ir.OOR,
	syntax.Xor: ir.OXOR,

	syntax.Mul:    ir.OMUL,
	syntax.Div:    ir.ODIV,
	syntax.Rem:    ir.OMOD,
	syntax.And:    ir.OAND,
	syntax.AndNot: ir.OANDNOT,
	syntax.Shl:    ir.OLSH,
	syntax.Shr:    ir.ORSH,
}

// error is called concurrently if files are parsed concurrently.
func (p *noder) error(err error) {
	p.err <- err.(syntax.Error)
}

// pragmas that are allowed in the std lib, but don't have
// a syntax.Pragma value (see lex.go) associated with them.
var allowedStdPragmas = map[string]bool{
	"go:cgo_export_static":  true,
	"go:cgo_export_dynamic": true,
	"go:cgo_import_static":  true,
	"go:cgo_import_dynamic": true,
	"go:cgo_ldflag":         true,
	"go:cgo_dynamic_linker": true,
	"go:embed":              true,
	"go:fix":                true,
	"go:generate":           true,
}

// *pragmas is the value stored in a syntax.pragmas during parsing.
type pragmas struct {
	Flag       ir.PragmaFlag // collected bits
	Pos        []pragmaPos   // position of each individual flag
	Embeds     []pragmaEmbed
	WasmImport *WasmImport
	WasmExport *WasmExport
}

// WasmImport stores metadata associated with the //go:wasmimport pragma
type WasmImport struct {
	Pos    syntax.Pos
	Module string
	Name   string
}

// WasmExport stores metadata associated with the //go:wasmexport pragma
type WasmExport struct {
	Pos  syntax.Pos
	Name string
}

type pragmaPos struct {
	Flag ir.PragmaFlag
	Pos  syntax.Pos
}

type pragmaEmbed struct {
	Pos      syntax.Pos
	Patterns []string
}

func (p *noder) checkUnusedDuringParse(pragma *pragmas) {
	for _, pos := range pragma.Pos {
		if pos.Flag&pragma.Flag != 0 {
			p.error(syntax.Error{Pos: pos.Pos, Msg: "misplaced compiler directive"})
		}
	}
	if len(pragma.Embeds) > 0 {
		for _, e := range pragma.Embeds {
			p.error(syntax.Error{Pos: e.Pos, Msg: "misplaced go:embed directive"})
		}
	}
	if pragma.WasmImport != nil {
		p.error(syntax.Error{Pos: pragma.WasmImport.Pos, Msg: "misplaced go:wasmimport directive"})
	}
	if pragma.WasmExport != nil {
		p.error(syntax.Error{Pos: pragma.WasmExport.Pos, Msg: "misplaced go:wasmexport directive"})
	}
}

// pragma is called concurrently if files are parsed concurrently.
func (p *noder) pragma(pos syntax.Pos, blankLine bool, text string, old syntax.Pragma) syntax.Pragma {
	pragma, _ := old.(*pragmas)
	if pragma == nil {
		pragma = new(pragmas)
	}

	if text == "" {
		// unused pragma; only called with old != nil.
		p.checkUnusedDuringParse(pragma)
		return nil
	}

	if strings.HasPrefix(text, "line ") {
		// line directives are handled by syntax package
		panic("unreachable")
	}

	if !blankLine {
		// directive must be on line by itself
		p.error(syntax.Error{Pos: pos, Msg: "misplaced compiler directive"})
		return pragma
	}

	switch {
	case strings.HasPrefix(text, "go:wasmimport "):
		f := strings.Fields(text)
		if len(f) != 3 {
			p.error(syntax.Error{Pos: pos, Msg: "usage: //go:wasmimport importmodule importname"})
			break
		}

		if buildcfg.GOARCH == "wasm" {
			// Only actually use them if we're compiling to WASM though.
			pragma.WasmImport = &WasmImport{
				Pos:    pos,
				Module: f[1],
				Name:   f[2],
			}
		}

	case strings.HasPrefix(text, "go:wasmexport "):
		f := strings.Fields(text)
		if len(f) != 2 {
			// TODO: maybe make the name optional? It was once mentioned on proposal 65199.
			p.error(syntax.Error{Pos: pos, Msg: "usage: //go:wasmexport exportname"})
			break
		}

		if buildcfg.GOARCH == "wasm" {
			// Only actually use them if we're compiling to WASM though.
			pragma.WasmExport = &WasmExport{
				Pos:  pos,
				Name: f[1],
			}
		}

	case strings.HasPrefix(text, "go:linkname "):
		f := strings.Fields(text)
		if !(2 <= len(f) && len(f) <= 3) {
			p.error(syntax.Error{Pos: pos, Msg: "usage: //go:linkname localname [linkname]"})
			break
		}
		// The second argument is optional. If omitted, we use
		// the default object symbol name for this and
		// linkname only serves to mark this symbol as
		// something that may be referenced via the object
		// symbol name from another package.
		var target string
		if len(f) == 3 {
			target = f[2]
		} else if base.Ctxt.Pkgpath != "" {
			// Use the default object symbol name if the
			// user didn't provide one.
			target = objabi.PathToPrefix(base.Ctxt.Pkgpath) + "." + f[1]
		} else {
			panic("missing pkgpath")
		}
		p.linknames = append(p.linknames, linkname{pos, f[1], target})

	case text == "go:embed", strings.HasPrefix(text, "go:embed "):
		args, err := parseGoEmbed(text[len("go:embed"):])
		if err != nil {
			p.error(syntax.Error{Pos: pos, Msg: err.Error()})
		}
		if len(args) == 0 {
			p.error(syntax.Error{Pos: pos, Msg: "usage: //go:embed pattern..."})
			break
		}
		pragma.Embeds = append(pragma.Embeds, pragmaEmbed{pos, args})

	case strings.HasPrefix(text, "go:cgo_import_dynamic "):
		// This is permitted for general use because Solaris
		// code relies on it in golang.org/x/sys/unix and others.
		fields := pragmaFields(text)
		if len(fields) >= 4 {
			lib := strings.Trim(fields[3], `"`)
			if lib != "" && !safeArg(lib) && !isCgoGeneratedFile(pos) {
				p.error(syntax.Error{Pos: pos, Msg: fmt.Sprintf("invalid library name %q in cgo_import_dynamic directive", lib)})
			}
			p.pragcgo(pos, text)
			pragma.Flag |= pragmaFlag("go:cgo_import_dynamic")
			break
		}
		fallthrough
	case strings.HasPrefix(text, "go:cgo_"):
		// For security, we disallow //go:cgo_* directives other
		// than cgo_import_dynamic outside cgo-generated files.
		// Exception: they are allowed in the standard library, for runtime and syscall.
		if !isCgoGeneratedFile(pos) && !base.Flag.Std {
			p.error(syntax.Error{Pos: pos, Msg: fmt.Sprintf("//%s only allowed in cgo-generated code", text)})
		}
		p.pragcgo(pos, text)
		fallthrough // because of //go:cgo_unsafe_args
	default:
		verb := text
		if i := strings.Index(text, " "); i >= 0 {
			verb = verb[:i]
		}
		flag := pragmaFlag(verb)
		const runtimePragmas = ir.Systemstack | ir.Nowritebarrier | ir.Nowritebarrierrec | ir.Yeswritebarrierrec
		if !base.Flag.CompilingRuntime && flag&runtimePragmas != 0 {
			p.error(syntax.Error{Pos: pos, Msg: fmt.Sprintf("//%s only allowed in runtime", verb)})
		}
		if flag == ir.UintptrKeepAlive && !base.Flag.Std {
			p.error(syntax.Error{Pos: pos, Msg: fmt.Sprintf("//%s is only allowed in the standard library", verb)})
		}
		if flag == 0 && !allowedStdPragmas[verb] && base.Flag.Std {
			p.error(syntax.Error{Pos: pos, Msg: fmt.Sprintf("//%s is not allowed in the standard library", verb)})
		}
		pragma.Flag |= flag
		pragma.Pos = append(pragma.Pos, pragmaPos{flag, pos})
	}

	return pragma
}

// isCgoGeneratedFile reports whether pos is in a file
// generated by cgo, which is to say a file with name
// beginning with "_cgo_". Such files are allowed to
// contain cgo directives, and for security reasons
// (primarily misuse of linker flags), other files are not.
// See golang.org/issue/23672.
// Note that cmd/go ignores files whose names start with underscore,
// so the only _cgo_ files we will see from cmd/go are generated by cgo.
// It's easy to bypass this check by calling the compiler directly;
// we only protect against uses by cmd/go.
func isCgoGeneratedFile(pos syntax.Pos) bool {
	// We need the absolute file, independent of //line directives,
	// so we call pos.Base().Pos().
	return strings.HasPrefix(filepath.Base(trimFilename(pos.Base().Pos().Base())), "_cgo_")
}

// safeArg reports whether arg is a "safe" command-line argument,
// meaning that when it appears in a command-line, it probably
// doesn't have some special meaning other than its own name.
// This is copied from SafeArg in cmd/go/internal/load/pkg.go.
func safeArg(name string) bool {
	if name == "" {
		return false
	}
	c := name[0]
	return '0' <= c && c <= '9' || 'A' <= c && c <= 'Z' || 'a' <= c && c <= 'z' || c == '.' || c == '_' || c == '/' || c >= utf8.RuneSelf
}

// parseGoEmbed parses the text following "//go:embed" to extract the glob patterns.
// It accepts unquoted space-separated patterns as well as double-quoted and back-quoted Go strings.
// go/build/read.go also processes these strings and contains similar logic.
func parseGoEmbed(args string) ([]string, error) {
	var list []string
	for args = strings.TrimSpace(args); args != ""; args = strings.TrimSpace(args) {
		var path string
	Switch:
		switch args[0] {
		default:
			i := len(args)
			for j, c := range args {
				if unicode.IsSpace(c) {
					i = j
					break
				}
			}
			path = args[:i]
			args = args[i:]

		case '`':
			i := strings.Index(args[1:], "`")
			if i < 0 {
				return nil, fmt.Errorf("invalid quoted string in //go:embed: %s", args)
			}
			path = args[1 : 1+i]
			args = args[1+i+1:]

		case '"':
			i := 1
			for ; i < len(args); i++ {
				if args[i] == '\\' {
					i++
					continue
				}
				if args[i] == '"' {
					q, err := strconv.Unquote(args[:i+1])
					if err != nil {
						return nil, fmt.Errorf("invalid quoted string in //go:embed: %s", args[:i+1])
					}
					path = q
					args = args[i+1:]
					break Switch
				}
			}
			if i >= len(args) {
				return nil, fmt.Errorf("invalid quoted string in //go:embed: %s", args)
			}
		}

		if args != "" {
			r, _ := utf8.DecodeRuneInString(args)
			if !unicode.IsSpace(r) {
				return nil, fmt.Errorf("invalid quoted string in //go:embed: %s", args)
			}
		}
		list = append(list, path)
	}
	return list, nil
}

// A function named init is a special case.
// It is called by the initialization before main is run.
// To make it unique within a package and also uncallable,
// the name, normally "pkg.init", is altered to "pkg.init.0".
var renameinitgen int

func Renameinit() *types.Sym {
	s := typecheck.LookupNum("init.", renameinitgen)
	renameinitgen++
	return s
}

func checkEmbed(decl *syntax.VarDecl, haveEmbed, withinFunc bool) error {
	switch {
	case !haveEmbed:
		return errors.New("go:embed requires import \"embed\" (or import _ \"embed\", if package is not used)")
	case len(decl.NameList) > 1:
		return errors.New("go:embed cannot apply to multiple vars")
	case decl.Values != nil:
		return errors.New("go:embed cannot apply to var with initializer")
	case decl.Type == nil:
		// Should not happen, since Values == nil now.
		return errors.New("go:embed cannot apply to var without type")
	case withinFunc:
		return errors.New("go:embed cannot apply to var inside func")
	case !types.AllowsGoVersion(1, 16):
		return fmt.Errorf("go:embed requires go1.16 or later (-lang was set to %s; check go.mod)", base.Flag.Lang)

	default:
		return nil
	}
}
