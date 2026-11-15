// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package load

import (
	"bytes"
	"fmt"
	"internal/logosfile"
	"path/filepath"
	"slices"
	"strings"

	"cmd/go/internal/cfg"
	"cmd/go/internal/fsys"
	"cmd/go/internal/logos/settings"
)

const (
	wasmOutputSuffix = ".wasm"
	wasmLoaderSuffix = ".loader.mjs"
)

const (
	goWasmExportDirective    = "//go:wasmexport "
	logosWasmExportDirective = "//logos:wasmexport "
)

// wantsWasmLoader reports the build-wide part of the loader decision (SPEC
// Sec 12.2, R-U1): the --wasm-loader flag is on, the build command is
// build, the target is GOARCH=wasm, and p is a main package. It reads the
// build configuration, never the module kind, so a Go main package and a
// loGo/s main package behave alike.
func wantsWasmLoader(p *Package) bool {
	return settings.Current().WasmLoader &&
		cfg.CmdName == "build" &&
		cfg.BuildContext.GOARCH == "wasm" &&
		p.Name == "main"
}

// isWasmFile reports whether out names a .wasm file.
func isWasmFile(out string) bool {
	return strings.HasSuffix(out, wasmOutputSuffix)
}

// EmitsWasmLoader reports at load time whether building p writes the host
// loader beside its output: the build-wide conditions hold and -o names a
// .wasm file. The load of p adds the implicit logos/rt import when it
// holds, so the exported helpers the loader calls are linked in. The
// default output name of a wasm main package carries no .wasm suffix, so
// a build without -o writes no loader.
func EmitsWasmLoader(p *Package) bool {
	return wantsWasmLoader(p) && isWasmFile(cfg.BuildO)
}

// WasmLoaderFor returns the loader file that goes beside the built output
// of p, and reports whether the build writes one: the build-wide
// conditions hold and the target of p is a .wasm file. The loader name is
// the target with its .wasm suffix replaced by .loader.mjs.
func WasmLoaderFor(p *Package) (string, bool) {
	if !wantsWasmLoader(p) || !isWasmFile(p.Target) {
		return "", false
	}
	return strings.TrimSuffix(p.Target, wasmOutputSuffix) + wasmLoaderSuffix, true
}

var loaderHelpers = []string{
	"_logos_scratch_alloc",
	"_logos_scratch_reset",
	"_logos_errmsg_ptr",
	"_logos_errmsg_len",
}

func pruneRedundantLogosRT(p *Package) {
	scan := graphWasmLoaderHelpers(p)
	if len(scan.offenders) > 0 {
		p.Error = &PackageError{Err: fmt.Errorf("%s; only the main module and %s may provide loader helpers", strings.Join(scan.offenders, "; "), logosRTModulePath)}
		p.Incomplete = true
		return
	}
	found := scan.found
	if len(found) == 0 {
		return
	}
	if len(found) == len(loaderHelpers) {
		p.Internal.Imports = dropLogosRTImport(p.Internal.Imports)
		p.Internal.CompiledImports = dropLogosRTCompiledImport(p.Internal.CompiledImports)
		return
	}
	var missing []string
	for _, name := range loaderHelpers {
		if !found[name] {
			missing = append(missing, name)
		}
	}
	p.Error = &PackageError{Err: fmt.Errorf("module graph provides some but not all wasm loader helpers: missing %s", strings.Join(missing, ", "))}
	p.Incomplete = true
}

const logosRTModulePath = "github.com/gosuda/loGos/rt"

type wasmLoaderHelperScan struct {
	found     map[string]bool
	offenders []string
}

func graphWasmLoaderHelpers(p *Package) wasmLoaderHelperScan {
	scan := wasmLoaderHelperScan{found: make(map[string]bool)}
	seen := make(map[*Package]bool)
	stack := append([]*Package{p}, p.Internal.Imports...)
	for len(stack) > 0 {
		q := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if q == nil || seen[q] {
			continue
		}
		seen[q] = true
		if q.ImportPath == LogosRTPkg || q.Standard {
			continue
		}
		helpers := collectPackageWasmLoaderHelpers(q)
		if wasmLoaderHelperAuthorized(q) {
			for _, name := range helpers {
				scan.found[name] = true
			}
		} else {
			for _, name := range helpers {
				scan.offenders = append(scan.offenders, fmt.Sprintf("wasm loader helper %s is exported by %s", name, q.ImportPath))
			}
		}
		stack = append(stack, q.Internal.Imports...)
	}
	slices.Sort(scan.offenders)
	return scan
}

func wasmLoaderHelperAuthorized(q *Package) bool {
	return q.ImportPath == logosRTModulePath || capIsMain(q)
}

func collectPackageWasmLoaderHelpers(q *Package) []string {
	var names []string
	seen := make(map[string]bool)
	for _, f := range q.GoFiles {
		data, err := fsys.ReadFile(filepath.Join(q.Dir, f))
		if err != nil {
			continue
		}
		for _, name := range fileWasmExportNames(f, data) {
			if slices.Contains(loaderHelpers, name) && !seen[name] {
				seen[name] = true
				names = append(names, name)
			}
		}
	}
	return names
}

func fileWasmExportNames(file string, data []byte) []string {
	s := wasmExportScanner{data: data, logos: logosfile.Is(file), blank: true}
	return s.scan()
}

type wasmExportScanner struct {
	data    []byte
	logos   bool
	pos     int
	blank   bool
	pending string
	names   []string
}

func (s *wasmExportScanner) scan() []string {
	for s.pos < len(s.data) {
		switch c := s.data[s.pos]; {
		case c == '/':
			s.slash()
		case isQuoteByte(c):
			s.quoted()
		case c == '$' && s.peek(1) == '"':
			s.interpStart()
		case isWordByte(c):
			s.word()
		default:
			s.plain(c)
		}
	}
	return s.names
}

func (s *wasmExportScanner) slash() {
	switch s.peek(1) {
	case '/':
		s.lineComment()
	case '*':
		s.blockComment()
	default:
		s.plain('/')
	}
}

func (s *wasmExportScanner) lineComment() {
	end := s.lineEnd()
	line := s.data[s.pos:end]
	s.pos = end
	if !s.blank || !s.exportDirective(line) {
		return
	}
	fields := strings.Fields(string(line))
	if len(fields) == 2 {
		s.pending = fields[1]
	}
}

func (s *wasmExportScanner) exportDirective(line []byte) bool {
	return bytes.HasPrefix(line, []byte(goWasmExportDirective)) ||
		s.logos && bytes.HasPrefix(line, []byte(logosWasmExportDirective))
}

func (s *wasmExportScanner) lineEnd() int {
	end := s.pos
	for end < len(s.data) && s.data[end] != '\n' {
		end++
	}
	return end
}

func (s *wasmExportScanner) blockComment() {
	s.pos += 2
	s.blank = false
	for s.pos < len(s.data) {
		if s.data[s.pos] == '*' && s.peek(1) == '/' {
			s.pos += 2
			return
		}
		s.pos++
	}
}

func (s *wasmExportScanner) word() {
	start := s.pos
	for s.pos < len(s.data) && isWordByte(s.data[s.pos]) {
		s.pos++
	}
	if s.pending != "" && string(s.data[start:s.pos]) == "func" && s.startsIdentifier() {
		s.names = append(s.names, s.pending)
	}
	s.code()
}

func (s *wasmExportScanner) quoted() {
	s.code()
	if s.data[s.pos] == '`' {
		s.rawString()
	} else {
		s.quotedString()
	}
}

func (s *wasmExportScanner) quotedString() {
	quote := s.data[s.pos]
	s.pos++
	for s.pos < len(s.data) {
		switch s.data[s.pos] {
		case quote:
			s.pos++
			return
		case '\\':
			s.pos += 2
		case '\n':
			return
		default:
			s.pos++
		}
	}
}

func (s *wasmExportScanner) rawString() {
	s.pos++
	for s.pos < len(s.data) {
		if s.data[s.pos] == '`' {
			s.pos++
			return
		}
		s.pos++
	}
}

func (s *wasmExportScanner) interpStart() {
	s.code()
	s.pos += 2
	s.interp()
}

func (s *wasmExportScanner) interp() {
	for s.pos < len(s.data) {
		switch s.data[s.pos] {
		case '"':
			s.pos++
			return
		case '\\':
			s.pos += 2
		case '\n':
			return
		case '{', '}':
			s.literalBrace()
		default:
			s.pos++
		}
	}
}

func (s *wasmExportScanner) literalBrace() {
	if s.data[s.pos] == '{' {
		if s.peek(1) == '{' {
			s.pos += 2
			return
		}
		s.pos++
		s.hole()
		return
	}
	if s.peek(1) == '}' {
		s.pos += 2
		return
	}
	s.pos++
}

func (s *wasmExportScanner) hole() {
	depth := 1
	for s.pos < len(s.data) && depth > 0 {
		depth = s.holeStep(depth)
	}
}

func (s *wasmExportScanner) holeStep(depth int) int {
	switch c := s.data[s.pos]; {
	case c == '{':
		s.pos++
		return depth + 1
	case c == '}':
		s.pos++
		return depth - 1
	case isQuoteByte(c):
		s.holeLiteral()
	case s.atComment():
		s.holeComment()
	case c == '\\':
		s.pos += 2
	default:
		s.pos++
	}
	return depth
}

func (s *wasmExportScanner) holeLiteral() {
	if s.data[s.pos] == '`' {
		s.rawString()
		return
	}
	s.quotedString()
}

func (s *wasmExportScanner) holeComment() {
	if s.peek(1) == '/' {
		s.skipLineComment()
		return
	}
	s.blockComment()
}

func (s *wasmExportScanner) atComment() bool {
	if s.data[s.pos] != '/' {
		return false
	}
	next := s.peek(1)
	return next == '/' || next == '*'
}

func (s *wasmExportScanner) skipLineComment() {
	for s.pos < len(s.data) && s.data[s.pos] != '\n' {
		s.pos++
	}
}

func (s *wasmExportScanner) plain(c byte) {
	if c == ' ' || c == '\t' || c == '\r' {
		s.pos++
		return
	}
	if c == '\n' {
		s.pos++
		s.blank = true
		return
	}
	s.code()
	s.pos++
}

func (s *wasmExportScanner) code() {
	s.pending = ""
	s.blank = false
}

func (s *wasmExportScanner) startsIdentifier() bool {
	ahead := wasmExportScanner{data: s.data, pos: s.pos}
	ahead.skipSpaceAndComments()
	return ahead.pos < len(ahead.data) && isLetterByte(ahead.data[ahead.pos])
}

func (s *wasmExportScanner) skipSpaceAndComments() {
	for s.pos < len(s.data) {
		switch {
		case isSpaceByte(s.data[s.pos]):
			s.pos++
		case s.atComment():
			s.holeComment()
		default:
			return
		}
	}
}

func isSpaceByte(c byte) bool {
	return c == ' ' || c == '\t' || c == '\r' || c == '\n'
}

func (s *wasmExportScanner) peek(n int) byte {
	if s.pos+n < len(s.data) {
		return s.data[s.pos+n]
	}
	return 0
}

func isQuoteByte(c byte) bool {
	return c == '"' || c == '\'' || c == '`'
}

func isWordByte(c byte) bool {
	return isLetterByte(c) || '0' <= c && c <= '9'
}

func isLetterByte(c byte) bool {
	return c == '_' || c >= 0x80 || 'a' <= c && c <= 'z' || 'A' <= c && c <= 'Z'
}

func dropLogosRTImport(pkgs []*Package) []*Package {
	kept := make([]*Package, 0, len(pkgs))
	for _, q := range pkgs {
		if q == nil || q.ImportPath == LogosRTPkg {
			continue
		}
		kept = append(kept, q)
	}
	return kept
}

func dropLogosRTCompiledImport(paths []string) []string {
	kept := make([]string, 0, len(paths))
	for _, s := range paths {
		if s == LogosRTPkg {
			continue
		}
		kept = append(kept, s)
	}
	return kept
}
