// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:generate go run mkbuiltin.go

package gc

import (
	"bufio"
	"bytes"
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/logopt"
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/types"
	"cmd/internal/bio"
	"cmd/internal/dwarf"
	"cmd/internal/goobj"
	"cmd/internal/obj"
	"cmd/internal/objabi"
	"cmd/internal/src"
	"flag"
	"fmt"
	"go/constant"
	"internal/goversion"
	"io"
	"io/ioutil"
	"log"
	"os"
	"path"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
)

func hidePanic() {
	if base.Debug.Panic == 0 && base.Errors() > 0 {
		// If we've already complained about things
		// in the program, don't bother complaining
		// about a panic too; let the user clean up
		// the code and try again.
		if err := recover(); err != nil {
			if err == "-h" {
				panic(err)
			}
			base.ErrorExit()
		}
	}
}

// timing data for compiler phases
var timings Timings

var nowritebarrierrecCheck *nowritebarrierrecChecker

// Main parses flags and Go source files specified in the command-line
// arguments, type-checks the parsed Go package, compiles functions to machine
// code, and finally writes the compiled package definition to disk.
func Main(archInit func(*Arch)) {
	timings.Start("fe", "init")

	defer hidePanic()

	archInit(&thearch)

	base.Ctxt = obj.Linknew(thearch.LinkArch)
	base.Ctxt.DiagFunc = base.Errorf
	base.Ctxt.DiagFlush = base.FlushErrors
	base.Ctxt.Bso = bufio.NewWriter(os.Stdout)

	// UseBASEntries is preferred because it shaves about 2% off build time, but LLDB, dsymutil, and dwarfdump
	// on Darwin don't support it properly, especially since macOS 10.14 (Mojave).  This is exposed as a flag
	// to allow testing with LLVM tools on Linux, and to help with reporting this bug to the LLVM project.
	// See bugs 31188 and 21945 (CLs 170638, 98075, 72371).
	base.Ctxt.UseBASEntries = base.Ctxt.Headtype != objabi.Hdarwin

	types.LocalPkg = types.NewPkg("", "")
	types.LocalPkg.Prefix = "\"\""

	// We won't know localpkg's height until after import
	// processing. In the mean time, set to MaxPkgHeight to ensure
	// height comparisons at least work until then.
	types.LocalPkg.Height = types.MaxPkgHeight

	// pseudo-package, for scoping
	types.BuiltinPkg = types.NewPkg("go.builtin", "") // TODO(gri) name this package go.builtin?
	types.BuiltinPkg.Prefix = "go.builtin"            // not go%2ebuiltin

	// pseudo-package, accessed by import "unsafe"
	unsafepkg = types.NewPkg("unsafe", "unsafe")

	// Pseudo-package that contains the compiler's builtin
	// declarations for package runtime. These are declared in a
	// separate package to avoid conflicts with package runtime's
	// actual declarations, which may differ intentionally but
	// insignificantly.
	Runtimepkg = types.NewPkg("go.runtime", "runtime")
	Runtimepkg.Prefix = "runtime"

	// pseudo-packages used in symbol tables
	itabpkg = types.NewPkg("go.itab", "go.itab")
	itabpkg.Prefix = "go.itab" // not go%2eitab

	itablinkpkg = types.NewPkg("go.itablink", "go.itablink")
	itablinkpkg.Prefix = "go.itablink" // not go%2eitablink

	trackpkg = types.NewPkg("go.track", "go.track")
	trackpkg.Prefix = "go.track" // not go%2etrack

	// pseudo-package used for map zero values
	mappkg = types.NewPkg("go.map", "go.map")
	mappkg.Prefix = "go.map"

	// pseudo-package used for methods with anonymous receivers
	gopkg = types.NewPkg("go", "")

	base.DebugSSA = ssa.PhaseOption
	base.ParseFlags()

	// Record flags that affect the build result. (And don't
	// record flags that don't, since that would cause spurious
	// changes in the binary.)
	recordFlags("B", "N", "l", "msan", "race", "shared", "dynlink", "dwarflocationlists", "dwarfbasentries", "smallframes", "spectre")

	if !enableTrace && base.Flag.LowerT {
		log.Fatalf("compiler not built with support for -t")
	}

	// Enable inlining (after recordFlags, to avoid recording the rewritten -l).  For now:
	//	default: inlining on.  (Flag.LowerL == 1)
	//	-l: inlining off  (Flag.LowerL == 0)
	//	-l=2, -l=3: inlining on again, with extra debugging (Flag.LowerL > 1)
	if base.Flag.LowerL <= 1 {
		base.Flag.LowerL = 1 - base.Flag.LowerL
	}

	if base.Flag.SmallFrames {
		maxStackVarSize = 128 * 1024
		maxImplicitStackVarSize = 16 * 1024
	}

	if base.Flag.Dwarf {
		base.Ctxt.DebugInfo = debuginfo
		base.Ctxt.GenAbstractFunc = genAbstractFunc
		base.Ctxt.DwFixups = obj.NewDwarfFixupTable(base.Ctxt)
	} else {
		// turn off inline generation if no dwarf at all
		base.Flag.GenDwarfInl = 0
		base.Ctxt.Flag_locationlists = false
	}
	if base.Ctxt.Flag_locationlists && len(base.Ctxt.Arch.DWARFRegisters) == 0 {
		log.Fatalf("location lists requested but register mapping not available on %v", base.Ctxt.Arch.Name)
	}

	checkLang()

	if base.Flag.SymABIs != "" {
		readSymABIs(base.Flag.SymABIs, base.Ctxt.Pkgpath)
	}

	if ispkgin(omit_pkgs) {
		base.Flag.Race = false
		base.Flag.MSan = false
	}

	thearch.LinkArch.Init(base.Ctxt)
	startProfile()
	if base.Flag.Race {
		racepkg = types.NewPkg("runtime/race", "")
	}
	if base.Flag.MSan {
		msanpkg = types.NewPkg("runtime/msan", "")
	}
	if base.Flag.Race || base.Flag.MSan {
		instrumenting = true
	}
	if base.Flag.Dwarf {
		dwarf.EnableLogging(base.Debug.DwarfInl != 0)
	}
	if base.Debug.SoftFloat != 0 {
		thearch.SoftFloat = true
	}

	if base.Flag.JSON != "" { // parse version,destination from json logging optimization.
		logopt.LogJsonOption(base.Flag.JSON)
	}

	ssaDump = os.Getenv("GOSSAFUNC")
	ssaDir = os.Getenv("GOSSADIR")
	if ssaDump != "" {
		if strings.HasSuffix(ssaDump, "+") {
			ssaDump = ssaDump[:len(ssaDump)-1]
			ssaDumpStdout = true
		}
		spl := strings.Split(ssaDump, ":")
		if len(spl) > 1 {
			ssaDump = spl[0]
			ssaDumpCFG = spl[1]
		}
	}

	trackScopes = base.Flag.Dwarf

	Widthptr = thearch.LinkArch.PtrSize
	Widthreg = thearch.LinkArch.RegSize

	// initialize types package
	// (we need to do this to break dependencies that otherwise
	// would lead to import cycles)
	initializeTypesPackage()

	dclcontext = ir.PEXTERN

	autogeneratedPos = makePos(src.NewFileBase("<autogenerated>", "<autogenerated>"), 1, 0)

	timings.Start("fe", "loadsys")
	loadsys()

	timings.Start("fe", "parse")
	lines := parseFiles(flag.Args())
	timings.Stop()
	timings.AddEvent(int64(lines), "lines")
	if base.Flag.G != 0 && base.Flag.G < 3 {
		// can only parse generic code for now
		base.ExitIfErrors()
		return
	}

	finishUniverse()

	recordPackageName()

	typecheckok = true

	// Process top-level declarations in phases.

	// Phase 1: const, type, and names and types of funcs.
	//   This will gather all the information about types
	//   and methods but doesn't depend on any of it.
	//
	//   We also defer type alias declarations until phase 2
	//   to avoid cycles like #18640.
	//   TODO(gri) Remove this again once we have a fix for #25838.

	// Don't use range--typecheck can add closures to xtop.
	timings.Start("fe", "typecheck", "top1")
	for i := 0; i < len(xtop); i++ {
		n := xtop[i]
		if op := n.Op(); op != ir.ODCL && op != ir.OAS && op != ir.OAS2 && (op != ir.ODCLTYPE || !n.Left().Name().Alias()) {
			xtop[i] = typecheck(n, ctxStmt)
		}
	}

	// Phase 2: Variable assignments.
	//   To check interface assignments, depends on phase 1.

	// Don't use range--typecheck can add closures to xtop.
	timings.Start("fe", "typecheck", "top2")
	for i := 0; i < len(xtop); i++ {
		n := xtop[i]
		if op := n.Op(); op == ir.ODCL || op == ir.OAS || op == ir.OAS2 || op == ir.ODCLTYPE && n.Left().Name().Alias() {
			xtop[i] = typecheck(n, ctxStmt)
		}
	}

	// Phase 3: Type check function bodies.
	// Don't use range--typecheck can add closures to xtop.
	timings.Start("fe", "typecheck", "func")
	var fcount int64
	for i := 0; i < len(xtop); i++ {
		n := xtop[i]
		if n.Op() == ir.ODCLFUNC {
			Curfn = n.(*ir.Func)
			decldepth = 1
			errorsBefore := base.Errors()
			typecheckslice(Curfn.Body().Slice(), ctxStmt)
			checkreturn(Curfn)
			if base.Errors() > errorsBefore {
				Curfn.PtrBody().Set(nil) // type errors; do not compile
			}
			// Now that we've checked whether n terminates,
			// we can eliminate some obviously dead code.
			deadcode(Curfn)
			fcount++
		}
	}
	// With all types checked, it's now safe to verify map keys. One single
	// check past phase 9 isn't sufficient, as we may exit with other errors
	// before then, thus skipping map key errors.
	checkMapKeys()
	base.ExitIfErrors()

	timings.AddEvent(fcount, "funcs")

	fninit(xtop)

	// Phase 4: Decide how to capture closed variables.
	// This needs to run before escape analysis,
	// because variables captured by value do not escape.
	timings.Start("fe", "capturevars")
	for _, n := range xtop {
		if n.Op() == ir.ODCLFUNC && n.Func().OClosure != nil {
			Curfn = n.(*ir.Func)
			capturevars(Curfn)
		}
	}
	capturevarscomplete = true
	Curfn = nil
	base.ExitIfErrors()

	// Phase 5: Inlining
	timings.Start("fe", "inlining")
	if base.Debug.TypecheckInl != 0 {
		// Typecheck imported function bodies if Debug.l > 1,
		// otherwise lazily when used or re-exported.
		for _, n := range importlist {
			if n.Inl != nil {
				typecheckinl(n)
			}
		}
		base.ExitIfErrors()
	}

	if base.Flag.LowerL != 0 {
		// Find functions that can be inlined and clone them before walk expands them.
		visitBottomUp(xtop, func(list []*ir.Func, recursive bool) {
			numfns := numNonClosures(list)
			for _, n := range list {
				if !recursive || numfns > 1 {
					// We allow inlining if there is no
					// recursion, or the recursion cycle is
					// across more than one function.
					caninl(n)
				} else {
					if base.Flag.LowerM > 1 {
						fmt.Printf("%v: cannot inline %v: recursive\n", ir.Line(n), n.Nname)
					}
				}
				inlcalls(n)
			}
		})
	}

	for _, n := range xtop {
		if n.Op() == ir.ODCLFUNC {
			devirtualize(n.(*ir.Func))
		}
	}
	Curfn = nil

	// Phase 6: Escape analysis.
	// Required for moving heap allocations onto stack,
	// which in turn is required by the closure implementation,
	// which stores the addresses of stack variables into the closure.
	// If the closure does not escape, it needs to be on the stack
	// or else the stack copier will not update it.
	// Large values are also moved off stack in escape analysis;
	// because large values may contain pointers, it must happen early.
	timings.Start("fe", "escapes")
	escapes(xtop)

	// Collect information for go:nowritebarrierrec
	// checking. This must happen before transformclosure.
	// We'll do the final check after write barriers are
	// inserted.
	if base.Flag.CompilingRuntime {
		nowritebarrierrecCheck = newNowritebarrierrecChecker()
	}

	// Phase 7: Transform closure bodies to properly reference captured variables.
	// This needs to happen before walk, because closures must be transformed
	// before walk reaches a call of a closure.
	timings.Start("fe", "xclosures")
	for _, n := range xtop {
		if n.Op() == ir.ODCLFUNC && n.Func().OClosure != nil {
			Curfn = n.(*ir.Func)
			transformclosure(Curfn)
		}
	}

	// Prepare for SSA compilation.
	// This must be before peekitabs, because peekitabs
	// can trigger function compilation.
	initssaconfig()

	// Just before compilation, compile itabs found on
	// the right side of OCONVIFACE so that methods
	// can be de-virtualized during compilation.
	Curfn = nil
	peekitabs()

	// Phase 8: Compile top level functions.
	// Don't use range--walk can add functions to xtop.
	timings.Start("be", "compilefuncs")
	fcount = 0
	for i := 0; i < len(xtop); i++ {
		n := xtop[i]
		if n.Op() == ir.ODCLFUNC {
			funccompile(n.(*ir.Func))
			fcount++
		}
	}
	timings.AddEvent(fcount, "funcs")

	compileFunctions()

	if nowritebarrierrecCheck != nil {
		// Write barriers are now known. Check the
		// call graph.
		nowritebarrierrecCheck.check()
		nowritebarrierrecCheck = nil
	}

	// Finalize DWARF inline routine DIEs, then explicitly turn off
	// DWARF inlining gen so as to avoid problems with generated
	// method wrappers.
	if base.Ctxt.DwFixups != nil {
		base.Ctxt.DwFixups.Finalize(base.Ctxt.Pkgpath, base.Debug.DwarfInl != 0)
		base.Ctxt.DwFixups = nil
		base.Flag.GenDwarfInl = 0
	}

	// Phase 9: Check external declarations.
	timings.Start("be", "externaldcls")
	for i, n := range externdcl {
		if n.Op() == ir.ONAME {
			externdcl[i] = typecheck(externdcl[i], ctxExpr)
		}
	}
	// Check the map keys again, since we typechecked the external
	// declarations.
	checkMapKeys()
	base.ExitIfErrors()

	// Write object data to disk.
	timings.Start("be", "dumpobj")
	dumpdata()
	base.Ctxt.NumberSyms()
	dumpobj()
	if base.Flag.AsmHdr != "" {
		dumpasmhdr()
	}

	// Check whether any of the functions we have compiled have gigantic stack frames.
	sort.Slice(largeStackFrames, func(i, j int) bool {
		return largeStackFrames[i].pos.Before(largeStackFrames[j].pos)
	})
	for _, large := range largeStackFrames {
		if large.callee != 0 {
			base.ErrorfAt(large.pos, "stack frame too large (>1GB): %d MB locals + %d MB args + %d MB callee", large.locals>>20, large.args>>20, large.callee>>20)
		} else {
			base.ErrorfAt(large.pos, "stack frame too large (>1GB): %d MB locals + %d MB args", large.locals>>20, large.args>>20)
		}
	}

	if len(funcStack) != 0 {
		base.Fatalf("funcStack is non-empty: %v", len(funcStack))
	}
	if len(compilequeue) != 0 {
		base.Fatalf("%d uncompiled functions", len(compilequeue))
	}

	logopt.FlushLoggedOpts(base.Ctxt, base.Ctxt.Pkgpath)
	base.ExitIfErrors()

	base.FlushErrors()
	timings.Stop()

	if base.Flag.Bench != "" {
		if err := writebench(base.Flag.Bench); err != nil {
			log.Fatalf("cannot write benchmark data: %v", err)
		}
	}
}

// numNonClosures returns the number of functions in list which are not closures.
func numNonClosures(list []*ir.Func) int {
	count := 0
	for _, fn := range list {
		if fn.OClosure == nil {
			count++
		}
	}
	return count
}

func writebench(filename string) error {
	f, err := os.OpenFile(filename, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0666)
	if err != nil {
		return err
	}

	var buf bytes.Buffer
	fmt.Fprintln(&buf, "commit:", objabi.Version)
	fmt.Fprintln(&buf, "goos:", runtime.GOOS)
	fmt.Fprintln(&buf, "goarch:", runtime.GOARCH)
	timings.Write(&buf, "BenchmarkCompile:"+base.Ctxt.Pkgpath+":")

	n, err := f.Write(buf.Bytes())
	if err != nil {
		return err
	}
	if n != buf.Len() {
		panic("bad writer")
	}

	return f.Close()
}

// symabiDefs and symabiRefs record the defined and referenced ABIs of
// symbols required by non-Go code. These are keyed by link symbol
// name, where the local package prefix is always `"".`
var symabiDefs, symabiRefs map[string]obj.ABI

// readSymABIs reads a symabis file that specifies definitions and
// references of text symbols by ABI.
//
// The symabis format is a set of lines, where each line is a sequence
// of whitespace-separated fields. The first field is a verb and is
// either "def" for defining a symbol ABI or "ref" for referencing a
// symbol using an ABI. For both "def" and "ref", the second field is
// the symbol name and the third field is the ABI name, as one of the
// named cmd/internal/obj.ABI constants.
func readSymABIs(file, myimportpath string) {
	data, err := ioutil.ReadFile(file)
	if err != nil {
		log.Fatalf("-symabis: %v", err)
	}

	symabiDefs = make(map[string]obj.ABI)
	symabiRefs = make(map[string]obj.ABI)

	localPrefix := ""
	if myimportpath != "" {
		// Symbols in this package may be written either as
		// "".X or with the package's import path already in
		// the symbol.
		localPrefix = objabi.PathToPrefix(myimportpath) + "."
	}

	for lineNum, line := range strings.Split(string(data), "\n") {
		lineNum++ // 1-based
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		parts := strings.Fields(line)
		switch parts[0] {
		case "def", "ref":
			// Parse line.
			if len(parts) != 3 {
				log.Fatalf(`%s:%d: invalid symabi: syntax is "%s sym abi"`, file, lineNum, parts[0])
			}
			sym, abistr := parts[1], parts[2]
			abi, valid := obj.ParseABI(abistr)
			if !valid {
				log.Fatalf(`%s:%d: invalid symabi: unknown abi "%s"`, file, lineNum, abistr)
			}

			// If the symbol is already prefixed with
			// myimportpath, rewrite it to start with ""
			// so it matches the compiler's internal
			// symbol names.
			if localPrefix != "" && strings.HasPrefix(sym, localPrefix) {
				sym = `"".` + sym[len(localPrefix):]
			}

			// Record for later.
			if parts[0] == "def" {
				symabiDefs[sym] = abi
			} else {
				symabiRefs[sym] = abi
			}
		default:
			log.Fatalf(`%s:%d: invalid symabi type "%s"`, file, lineNum, parts[0])
		}
	}
}

func arsize(b *bufio.Reader, name string) int {
	var buf [ArhdrSize]byte
	if _, err := io.ReadFull(b, buf[:]); err != nil {
		return -1
	}
	aname := strings.Trim(string(buf[0:16]), " ")
	if !strings.HasPrefix(aname, name) {
		return -1
	}
	asize := strings.Trim(string(buf[48:58]), " ")
	i, _ := strconv.Atoi(asize)
	return i
}

func isDriveLetter(b byte) bool {
	return 'a' <= b && b <= 'z' || 'A' <= b && b <= 'Z'
}

// is this path a local name? begins with ./ or ../ or /
func islocalname(name string) bool {
	return strings.HasPrefix(name, "/") ||
		runtime.GOOS == "windows" && len(name) >= 3 && isDriveLetter(name[0]) && name[1] == ':' && name[2] == '/' ||
		strings.HasPrefix(name, "./") || name == "." ||
		strings.HasPrefix(name, "../") || name == ".."
}

func findpkg(name string) (file string, ok bool) {
	if islocalname(name) {
		if base.Flag.NoLocalImports {
			return "", false
		}

		if base.Flag.Cfg.PackageFile != nil {
			file, ok = base.Flag.Cfg.PackageFile[name]
			return file, ok
		}

		// try .a before .6.  important for building libraries:
		// if there is an array.6 in the array.a library,
		// want to find all of array.a, not just array.6.
		file = fmt.Sprintf("%s.a", name)
		if _, err := os.Stat(file); err == nil {
			return file, true
		}
		file = fmt.Sprintf("%s.o", name)
		if _, err := os.Stat(file); err == nil {
			return file, true
		}
		return "", false
	}

	// local imports should be canonicalized already.
	// don't want to see "encoding/../encoding/base64"
	// as different from "encoding/base64".
	if q := path.Clean(name); q != name {
		base.Errorf("non-canonical import path %q (should be %q)", name, q)
		return "", false
	}

	if base.Flag.Cfg.PackageFile != nil {
		file, ok = base.Flag.Cfg.PackageFile[name]
		return file, ok
	}

	for _, dir := range base.Flag.Cfg.ImportDirs {
		file = fmt.Sprintf("%s/%s.a", dir, name)
		if _, err := os.Stat(file); err == nil {
			return file, true
		}
		file = fmt.Sprintf("%s/%s.o", dir, name)
		if _, err := os.Stat(file); err == nil {
			return file, true
		}
	}

	if objabi.GOROOT != "" {
		suffix := ""
		suffixsep := ""
		if base.Flag.InstallSuffix != "" {
			suffixsep = "_"
			suffix = base.Flag.InstallSuffix
		} else if base.Flag.Race {
			suffixsep = "_"
			suffix = "race"
		} else if base.Flag.MSan {
			suffixsep = "_"
			suffix = "msan"
		}

		file = fmt.Sprintf("%s/pkg/%s_%s%s%s/%s.a", objabi.GOROOT, objabi.GOOS, objabi.GOARCH, suffixsep, suffix, name)
		if _, err := os.Stat(file); err == nil {
			return file, true
		}
		file = fmt.Sprintf("%s/pkg/%s_%s%s%s/%s.o", objabi.GOROOT, objabi.GOOS, objabi.GOARCH, suffixsep, suffix, name)
		if _, err := os.Stat(file); err == nil {
			return file, true
		}
	}

	return "", false
}

// loadsys loads the definitions for the low-level runtime functions,
// so that the compiler can generate calls to them,
// but does not make them visible to user code.
func loadsys() {
	types.Block = 1

	inimport = true
	typecheckok = true

	typs := runtimeTypes()
	for _, d := range &runtimeDecls {
		sym := Runtimepkg.Lookup(d.name)
		typ := typs[d.typ]
		switch d.tag {
		case funcTag:
			importfunc(Runtimepkg, src.NoXPos, sym, typ)
		case varTag:
			importvar(Runtimepkg, src.NoXPos, sym, typ)
		default:
			base.Fatalf("unhandled declaration tag %v", d.tag)
		}
	}

	typecheckok = false
	inimport = false
}

// myheight tracks the local package's height based on packages
// imported so far.
var myheight int

func importfile(f constant.Value) *types.Pkg {
	if f.Kind() != constant.String {
		base.Errorf("import path must be a string")
		return nil
	}

	path_ := constant.StringVal(f)
	if len(path_) == 0 {
		base.Errorf("import path is empty")
		return nil
	}

	if isbadimport(path_, false) {
		return nil
	}

	// The package name main is no longer reserved,
	// but we reserve the import path "main" to identify
	// the main package, just as we reserve the import
	// path "math" to identify the standard math package.
	if path_ == "main" {
		base.Errorf("cannot import \"main\"")
		base.ErrorExit()
	}

	if base.Ctxt.Pkgpath != "" && path_ == base.Ctxt.Pkgpath {
		base.Errorf("import %q while compiling that package (import cycle)", path_)
		base.ErrorExit()
	}

	if mapped, ok := base.Flag.Cfg.ImportMap[path_]; ok {
		path_ = mapped
	}

	if path_ == "unsafe" {
		return unsafepkg
	}

	if islocalname(path_) {
		if path_[0] == '/' {
			base.Errorf("import path cannot be absolute path")
			return nil
		}

		prefix := base.Ctxt.Pathname
		if base.Flag.D != "" {
			prefix = base.Flag.D
		}
		path_ = path.Join(prefix, path_)

		if isbadimport(path_, true) {
			return nil
		}
	}

	file, found := findpkg(path_)
	if !found {
		base.Errorf("can't find import: %q", path_)
		base.ErrorExit()
	}

	importpkg := types.NewPkg(path_, "")
	if importpkg.Imported {
		return importpkg
	}

	importpkg.Imported = true

	imp, err := bio.Open(file)
	if err != nil {
		base.Errorf("can't open import: %q: %v", path_, err)
		base.ErrorExit()
	}
	defer imp.Close()

	// check object header
	p, err := imp.ReadString('\n')
	if err != nil {
		base.Errorf("import %s: reading input: %v", file, err)
		base.ErrorExit()
	}

	if p == "!<arch>\n" { // package archive
		// package export block should be first
		sz := arsize(imp.Reader, "__.PKGDEF")
		if sz <= 0 {
			base.Errorf("import %s: not a package file", file)
			base.ErrorExit()
		}
		p, err = imp.ReadString('\n')
		if err != nil {
			base.Errorf("import %s: reading input: %v", file, err)
			base.ErrorExit()
		}
	}

	if !strings.HasPrefix(p, "go object ") {
		base.Errorf("import %s: not a go object file: %s", file, p)
		base.ErrorExit()
	}
	q := fmt.Sprintf("%s %s %s %s\n", objabi.GOOS, objabi.GOARCH, objabi.Version, objabi.Expstring())
	if p[10:] != q {
		base.Errorf("import %s: object is [%s] expected [%s]", file, p[10:], q)
		base.ErrorExit()
	}

	// process header lines
	for {
		p, err = imp.ReadString('\n')
		if err != nil {
			base.Errorf("import %s: reading input: %v", file, err)
			base.ErrorExit()
		}
		if p == "\n" {
			break // header ends with blank line
		}
	}

	// Expect $$B\n to signal binary import format.

	// look for $$
	var c byte
	for {
		c, err = imp.ReadByte()
		if err != nil {
			break
		}
		if c == '$' {
			c, err = imp.ReadByte()
			if c == '$' || err != nil {
				break
			}
		}
	}

	// get character after $$
	if err == nil {
		c, _ = imp.ReadByte()
	}

	var fingerprint goobj.FingerprintType
	switch c {
	case '\n':
		base.Errorf("cannot import %s: old export format no longer supported (recompile library)", path_)
		return nil

	case 'B':
		if base.Debug.Export != 0 {
			fmt.Printf("importing %s (%s)\n", path_, file)
		}
		imp.ReadByte() // skip \n after $$B

		c, err = imp.ReadByte()
		if err != nil {
			base.Errorf("import %s: reading input: %v", file, err)
			base.ErrorExit()
		}

		// Indexed format is distinguished by an 'i' byte,
		// whereas previous export formats started with 'c', 'd', or 'v'.
		if c != 'i' {
			base.Errorf("import %s: unexpected package format byte: %v", file, c)
			base.ErrorExit()
		}
		fingerprint = iimport(importpkg, imp)

	default:
		base.Errorf("no import in %q", path_)
		base.ErrorExit()
	}

	// assume files move (get installed) so don't record the full path
	if base.Flag.Cfg.PackageFile != nil {
		// If using a packageFile map, assume path_ can be recorded directly.
		base.Ctxt.AddImport(path_, fingerprint)
	} else {
		// For file "/Users/foo/go/pkg/darwin_amd64/math.a" record "math.a".
		base.Ctxt.AddImport(file[len(file)-len(path_)-len(".a"):], fingerprint)
	}

	if importpkg.Height >= myheight {
		myheight = importpkg.Height + 1
	}

	return importpkg
}

func pkgnotused(lineno src.XPos, path string, name string) {
	// If the package was imported with a name other than the final
	// import path element, show it explicitly in the error message.
	// Note that this handles both renamed imports and imports of
	// packages containing unconventional package declarations.
	// Note that this uses / always, even on Windows, because Go import
	// paths always use forward slashes.
	elem := path
	if i := strings.LastIndex(elem, "/"); i >= 0 {
		elem = elem[i+1:]
	}
	if name == "" || elem == name {
		base.ErrorfAt(lineno, "imported and not used: %q", path)
	} else {
		base.ErrorfAt(lineno, "imported and not used: %q as %s", path, name)
	}
}

func mkpackage(pkgname string) {
	if types.LocalPkg.Name == "" {
		if pkgname == "_" {
			base.Errorf("invalid package name _")
		}
		types.LocalPkg.Name = pkgname
	} else {
		if pkgname != types.LocalPkg.Name {
			base.Errorf("package %s; expected %s", pkgname, types.LocalPkg.Name)
		}
	}
}

func clearImports() {
	type importedPkg struct {
		pos  src.XPos
		path string
		name string
	}
	var unused []importedPkg

	for _, s := range types.LocalPkg.Syms {
		n := ir.AsNode(s.Def)
		if n == nil {
			continue
		}
		if n.Op() == ir.OPACK {
			// throw away top-level package name left over
			// from previous file.
			// leave s->block set to cause redeclaration
			// errors if a conflicting top-level name is
			// introduced by a different file.
			p := n.(*ir.PkgName)
			if !p.Used && base.SyntaxErrors() == 0 {
				unused = append(unused, importedPkg{p.Pos(), p.Pkg.Path, s.Name})
			}
			s.Def = nil
			continue
		}
		if IsAlias(s) {
			// throw away top-level name left over
			// from previous import . "x"
			if name := n.Name(); name != nil && name.PkgName != nil && !name.PkgName.Used && base.SyntaxErrors() == 0 {
				unused = append(unused, importedPkg{name.PkgName.Pos(), name.PkgName.Pkg.Path, ""})
				name.PkgName.Used = true
			}
			s.Def = nil
			continue
		}
	}

	sort.Slice(unused, func(i, j int) bool { return unused[i].pos.Before(unused[j].pos) })
	for _, pkg := range unused {
		pkgnotused(pkg.pos, pkg.path, pkg.name)
	}
}

func IsAlias(sym *types.Sym) bool {
	return sym.Def != nil && sym.Def.Sym() != sym
}

// recordFlags records the specified command-line flags to be placed
// in the DWARF info.
func recordFlags(flags ...string) {
	if base.Ctxt.Pkgpath == "" {
		// We can't record the flags if we don't know what the
		// package name is.
		return
	}

	type BoolFlag interface {
		IsBoolFlag() bool
	}
	type CountFlag interface {
		IsCountFlag() bool
	}
	var cmd bytes.Buffer
	for _, name := range flags {
		f := flag.Lookup(name)
		if f == nil {
			continue
		}
		getter := f.Value.(flag.Getter)
		if getter.String() == f.DefValue {
			// Flag has default value, so omit it.
			continue
		}
		if bf, ok := f.Value.(BoolFlag); ok && bf.IsBoolFlag() {
			val, ok := getter.Get().(bool)
			if ok && val {
				fmt.Fprintf(&cmd, " -%s", f.Name)
				continue
			}
		}
		if cf, ok := f.Value.(CountFlag); ok && cf.IsCountFlag() {
			val, ok := getter.Get().(int)
			if ok && val == 1 {
				fmt.Fprintf(&cmd, " -%s", f.Name)
				continue
			}
		}
		fmt.Fprintf(&cmd, " -%s=%v", f.Name, getter.Get())
	}

	if cmd.Len() == 0 {
		return
	}
	s := base.Ctxt.Lookup(dwarf.CUInfoPrefix + "producer." + base.Ctxt.Pkgpath)
	s.Type = objabi.SDWARFCUINFO
	// Sometimes (for example when building tests) we can link
	// together two package main archives. So allow dups.
	s.Set(obj.AttrDuplicateOK, true)
	base.Ctxt.Data = append(base.Ctxt.Data, s)
	s.P = cmd.Bytes()[1:]
}

// recordPackageName records the name of the package being
// compiled, so that the linker can save it in the compile unit's DIE.
func recordPackageName() {
	s := base.Ctxt.Lookup(dwarf.CUInfoPrefix + "packagename." + base.Ctxt.Pkgpath)
	s.Type = objabi.SDWARFCUINFO
	// Sometimes (for example when building tests) we can link
	// together two package main archives. So allow dups.
	s.Set(obj.AttrDuplicateOK, true)
	base.Ctxt.Data = append(base.Ctxt.Data, s)
	s.P = []byte(types.LocalPkg.Name)
}

// currentLang returns the current language version.
func currentLang() string {
	return fmt.Sprintf("go1.%d", goversion.Version)
}

// goVersionRE is a regular expression that matches the valid
// arguments to the -lang flag.
var goVersionRE = regexp.MustCompile(`^go([1-9][0-9]*)\.(0|[1-9][0-9]*)$`)

// A lang is a language version broken into major and minor numbers.
type lang struct {
	major, minor int
}

// langWant is the desired language version set by the -lang flag.
// If the -lang flag is not set, this is the zero value, meaning that
// any language version is supported.
var langWant lang

// langSupported reports whether language version major.minor is
// supported in a particular package.
func langSupported(major, minor int, pkg *types.Pkg) bool {
	if pkg == nil {
		// TODO(mdempsky): Set Pkg for local types earlier.
		pkg = types.LocalPkg
	}
	if pkg != types.LocalPkg {
		// Assume imported packages passed type-checking.
		return true
	}

	if langWant.major == 0 && langWant.minor == 0 {
		return true
	}
	return langWant.major > major || (langWant.major == major && langWant.minor >= minor)
}

// checkLang verifies that the -lang flag holds a valid value, and
// exits if not. It initializes data used by langSupported.
func checkLang() {
	if base.Flag.Lang == "" {
		return
	}

	var err error
	langWant, err = parseLang(base.Flag.Lang)
	if err != nil {
		log.Fatalf("invalid value %q for -lang: %v", base.Flag.Lang, err)
	}

	if def := currentLang(); base.Flag.Lang != def {
		defVers, err := parseLang(def)
		if err != nil {
			log.Fatalf("internal error parsing default lang %q: %v", def, err)
		}
		if langWant.major > defVers.major || (langWant.major == defVers.major && langWant.minor > defVers.minor) {
			log.Fatalf("invalid value %q for -lang: max known version is %q", base.Flag.Lang, def)
		}
	}
}

// parseLang parses a -lang option into a langVer.
func parseLang(s string) (lang, error) {
	matches := goVersionRE.FindStringSubmatch(s)
	if matches == nil {
		return lang{}, fmt.Errorf(`should be something like "go1.12"`)
	}
	major, err := strconv.Atoi(matches[1])
	if err != nil {
		return lang{}, err
	}
	minor, err := strconv.Atoi(matches[2])
	if err != nil {
		return lang{}, err
	}
	return lang{major: major, minor: minor}, nil
}

func initializeTypesPackage() {
	types.Widthptr = Widthptr
	types.Dowidth = dowidth
	types.TypeLinkSym = func(t *types.Type) *obj.LSym {
		return typenamesym(t).Linksym()
	}

	initUniverse()
}
