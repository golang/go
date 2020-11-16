// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"runtime"
	"strconv"
	"strings"

	"cmd/compile/internal/logopt"
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/types"
	"cmd/internal/dwarf"
	"cmd/internal/obj"
	"cmd/internal/objabi"
	"cmd/internal/sys"
)

func usage() {
	fmt.Fprintf(os.Stderr, "usage: compile [options] file.go...\n")
	objabi.Flagprint(os.Stderr)
	Exit(2)
}

var Flag Flags

// gc debug flags
type Flags struct {
	Percent, B, C, E,
	K, L, N, S,
	W, LowerE, LowerH, LowerJ,
	LowerL, LowerM, LowerR, LowerW int
	CompilingRuntime bool
	Std              bool
	D                string
	AsmHdr           string
	BuildID          string
	LowerC           int
	Complete         bool
	LowerD           string
	Dwarf            bool
	GenDwarfInl      int
	InstallSuffix    string
	Lang             string
	LinkObj          string
	Live             int
	MSan             bool
	NoLocalImports   bool
	LowerO           string
	Pack             bool
	Race             bool
	Spectre          string
	LowerT           bool
	TrimPath         string
	WB               bool
	Shared           bool
	Dynlink          bool
	GoVersion        string
	SymABIs          string
	CPUProfile       string
	MemProfile       string
	TraceProfile     string
	BlockProfile     string
	MutexProfile     string
	Bench            string
	SmallFrames      bool
	JSON             string

	Cfg struct {
		Embed struct {
			Patterns map[string][]string
			Files    map[string]string
		}
		ImportDirs   []string
		ImportMap    map[string]string
		PackageFile  map[string]string
		SpectreIndex bool
	}
}

func ParseFlags() {
	Wasm := objabi.GOARCH == "wasm"

	// Whether the limit for stack-allocated objects is much smaller than normal.
	// This can be helpful for diagnosing certain causes of GC latency. See #27732.
	Flag.SmallFrames = false
	Flag.JSON = ""

	flag.BoolVar(&Flag.CompilingRuntime, "+", false, "compiling runtime")
	flag.BoolVar(&Flag.Std, "std", false, "compiling standard library")
	flag.StringVar(&Flag.D, "D", "", "set relative `path` for local imports")

	objabi.Flagcount("%", "debug non-static initializers", &Flag.Percent)
	objabi.Flagcount("B", "disable bounds checking", &Flag.B)
	objabi.Flagcount("C", "disable printing of columns in error messages", &Flag.C)
	objabi.Flagcount("E", "debug symbol export", &Flag.E)
	objabi.Flagcount("K", "debug missing line numbers", &Flag.K)
	objabi.Flagcount("L", "show full file names in error messages", &Flag.L)
	objabi.Flagcount("N", "disable optimizations", &Flag.N)
	objabi.Flagcount("S", "print assembly listing", &Flag.S)
	objabi.Flagcount("W", "debug parse tree after type checking", &Flag.W)
	objabi.Flagcount("e", "no limit on number of errors reported", &Flag.LowerE)
	objabi.Flagcount("h", "halt on error", &Flag.LowerH)
	objabi.Flagcount("j", "debug runtime-initialized variables", &Flag.LowerJ)
	objabi.Flagcount("l", "disable inlining", &Flag.LowerL)
	objabi.Flagcount("m", "print optimization decisions", &Flag.LowerM)
	objabi.Flagcount("r", "debug generated wrappers", &Flag.LowerR)
	objabi.Flagcount("w", "debug type checking", &Flag.LowerW)

	objabi.Flagfn1("I", "add `directory` to import search path", addImportDir)
	objabi.AddVersionFlag() // -V
	flag.StringVar(&Flag.AsmHdr, "asmhdr", "", "write assembly header to `file`")
	flag.StringVar(&Flag.BuildID, "buildid", "", "record `id` as the build id in the export metadata")
	flag.IntVar(&Flag.LowerC, "c", 1, "concurrency during compilation, 1 means no concurrency")
	flag.BoolVar(&Flag.Complete, "complete", false, "compiling complete package (no C or assembly)")
	flag.StringVar(&Flag.LowerD, "d", "", "print debug information about items in `list`; try -d help")
	flag.BoolVar(&Flag.Dwarf, "dwarf", !Wasm, "generate DWARF symbols")
	flag.BoolVar(&Ctxt.Flag_locationlists, "dwarflocationlists", true, "add location lists to DWARF in optimized mode")
	flag.IntVar(&Flag.GenDwarfInl, "gendwarfinl", 2, "generate DWARF inline info records")
	objabi.Flagfn1("embedcfg", "read go:embed configuration from `file`", readEmbedCfg)
	objabi.Flagfn1("importmap", "add `definition` of the form source=actual to import map", addImportMap)
	objabi.Flagfn1("importcfg", "read import configuration from `file`", readImportCfg)
	flag.StringVar(&Flag.InstallSuffix, "installsuffix", "", "set pkg directory `suffix`")
	flag.StringVar(&Flag.Lang, "lang", "", "release to compile for")
	flag.StringVar(&Flag.LinkObj, "linkobj", "", "write linker-specific object to `file`")
	objabi.Flagcount("live", "debug liveness analysis", &Flag.Live)
	if sys.MSanSupported(objabi.GOOS, objabi.GOARCH) {
		flag.BoolVar(&Flag.MSan, "msan", false, "build code compatible with C/C++ memory sanitizer")
	}
	flag.BoolVar(&Flag.NoLocalImports, "nolocalimports", false, "reject local (relative) imports")
	flag.StringVar(&Flag.LowerO, "o", "", "write output to `file`")
	flag.StringVar(&Ctxt.Pkgpath, "p", "", "set expected package import `path`")
	flag.BoolVar(&Flag.Pack, "pack", false, "write to file.a instead of file.o")
	if sys.RaceDetectorSupported(objabi.GOOS, objabi.GOARCH) {
		flag.BoolVar(&Flag.Race, "race", false, "enable race detector")
	}
	flag.StringVar(&Flag.Spectre, "spectre", Flag.Spectre, "enable spectre mitigations in `list` (all, index, ret)")
	if enableTrace {
		flag.BoolVar(&Flag.LowerT, "t", false, "trace type-checking")
	}
	flag.StringVar(&Flag.TrimPath, "trimpath", "", "remove `prefix` from recorded source file paths")
	flag.BoolVar(&Ctxt.Debugvlog, "v", false, "increase debug verbosity")
	flag.BoolVar(&Flag.WB, "wb", true, "enable write barrier")
	if supportsDynlink(thearch.LinkArch.Arch) {
		flag.BoolVar(&Flag.Shared, "shared", false, "generate code that can be linked into a shared library")
		flag.BoolVar(&Flag.Dynlink, "dynlink", false, "support references to Go symbols defined in other shared libraries")
		flag.BoolVar(&Ctxt.Flag_linkshared, "linkshared", false, "generate code that will be linked against Go shared libraries")
	}
	flag.StringVar(&Flag.CPUProfile, "cpuprofile", "", "write cpu profile to `file`")
	flag.StringVar(&Flag.MemProfile, "memprofile", "", "write memory profile to `file`")
	flag.Int64Var(&memprofilerate, "memprofilerate", 0, "set runtime.MemProfileRate to `rate`")
	flag.StringVar(&Flag.GoVersion, "goversion", "", "required version of the runtime")
	flag.StringVar(&Flag.SymABIs, "symabis", "", "read symbol ABIs from `file`")
	flag.StringVar(&Flag.TraceProfile, "traceprofile", "", "write an execution trace to `file`")
	flag.StringVar(&Flag.BlockProfile, "blockprofile", "", "write block profile to `file`")
	flag.StringVar(&Flag.MutexProfile, "mutexprofile", "", "write mutex profile to `file`")
	flag.StringVar(&Flag.Bench, "bench", "", "append benchmark times to `file`")
	flag.BoolVar(&Flag.SmallFrames, "smallframes", false, "reduce the size limit for stack allocated objects")
	flag.BoolVar(&Ctxt.UseBASEntries, "dwarfbasentries", Ctxt.UseBASEntries, "use base address selection entries in DWARF")
	flag.StringVar(&Flag.JSON, "json", "", "version,destination for JSON compiler/optimizer logging")

	objabi.Flagparse(usage)

	for _, f := range strings.Split(Flag.Spectre, ",") {
		f = strings.TrimSpace(f)
		switch f {
		default:
			log.Fatalf("unknown setting -spectre=%s", f)
		case "":
			// nothing
		case "all":
			Flag.Cfg.SpectreIndex = true
			Ctxt.Retpoline = true
		case "index":
			Flag.Cfg.SpectreIndex = true
		case "ret":
			Ctxt.Retpoline = true
		}
	}

	if Flag.Cfg.SpectreIndex {
		switch objabi.GOARCH {
		case "amd64":
			// ok
		default:
			log.Fatalf("GOARCH=%s does not support -spectre=index", objabi.GOARCH)
		}
	}

	// Record flags that affect the build result. (And don't
	// record flags that don't, since that would cause spurious
	// changes in the binary.)
	recordFlags("B", "N", "l", "msan", "race", "shared", "dynlink", "dwarflocationlists", "dwarfbasentries", "smallframes", "spectre")

	if Flag.SmallFrames {
		maxStackVarSize = 128 * 1024
		maxImplicitStackVarSize = 16 * 1024
	}

	Ctxt.Flag_shared = Flag.Dynlink || Flag.Shared
	Ctxt.Flag_dynlink = Flag.Dynlink
	Ctxt.Flag_optimize = Flag.N == 0

	Ctxt.Debugasm = Flag.S
	if Flag.Dwarf {
		Ctxt.DebugInfo = debuginfo
		Ctxt.GenAbstractFunc = genAbstractFunc
		Ctxt.DwFixups = obj.NewDwarfFixupTable(Ctxt)
	} else {
		// turn off inline generation if no dwarf at all
		Flag.GenDwarfInl = 0
		Ctxt.Flag_locationlists = false
	}

	if flag.NArg() < 1 && Flag.LowerD != "help" && Flag.LowerD != "ssa/help" {
		usage()
	}

	if Flag.GoVersion != "" && Flag.GoVersion != runtime.Version() {
		fmt.Printf("compile: version %q does not match go tool version %q\n", runtime.Version(), Flag.GoVersion)
		Exit(2)
	}

	checkLang()

	if Flag.SymABIs != "" {
		readSymABIs(Flag.SymABIs, Ctxt.Pkgpath)
	}

	thearch.LinkArch.Init(Ctxt)

	if Flag.LowerO == "" {
		p := flag.Arg(0)
		if i := strings.LastIndex(p, "/"); i >= 0 {
			p = p[i+1:]
		}
		if runtime.GOOS == "windows" {
			if i := strings.LastIndex(p, `\`); i >= 0 {
				p = p[i+1:]
			}
		}
		if i := strings.LastIndex(p, "."); i >= 0 {
			p = p[:i]
		}
		suffix := ".o"
		if Flag.Pack {
			suffix = ".a"
		}
		Flag.LowerO = p + suffix
	}

	startProfile()

	if Flag.Race && Flag.MSan {
		log.Fatal("cannot use both -race and -msan")
	}
	if Flag.Race || Flag.MSan {
		// -race and -msan imply -d=checkptr for now.
		Debug_checkptr = 1
	}
	if ispkgin(omit_pkgs) {
		Flag.Race = false
		Flag.MSan = false
	}
	if Flag.Race {
		racepkg = types.NewPkg("runtime/race", "")
	}
	if Flag.MSan {
		msanpkg = types.NewPkg("runtime/msan", "")
	}
	if Flag.Race || Flag.MSan {
		instrumenting = true
	}

	if Flag.CompilingRuntime && Flag.N != 0 {
		log.Fatal("cannot disable optimizations while compiling runtime")
	}
	if Flag.LowerC < 1 {
		log.Fatalf("-c must be at least 1, got %d", Flag.LowerC)
	}
	if Flag.LowerC > 1 && !concurrentBackendAllowed() {
		log.Fatalf("cannot use concurrent backend compilation with provided flags; invoked as %v", os.Args)
	}
	if Ctxt.Flag_locationlists && len(Ctxt.Arch.DWARFRegisters) == 0 {
		log.Fatalf("location lists requested but register mapping not available on %v", Ctxt.Arch.Name)
	}

	// parse -d argument
	if Flag.LowerD != "" {
	Split:
		for _, name := range strings.Split(Flag.LowerD, ",") {
			if name == "" {
				continue
			}
			// display help about the -d option itself and quit
			if name == "help" {
				fmt.Print(debugHelpHeader)
				maxLen := len("ssa/help")
				for _, t := range debugtab {
					if len(t.name) > maxLen {
						maxLen = len(t.name)
					}
				}
				for _, t := range debugtab {
					fmt.Printf("\t%-*s\t%s\n", maxLen, t.name, t.help)
				}
				// ssa options have their own help
				fmt.Printf("\t%-*s\t%s\n", maxLen, "ssa/help", "print help about SSA debugging")
				fmt.Print(debugHelpFooter)
				os.Exit(0)
			}
			val, valstring, haveInt := 1, "", true
			if i := strings.IndexAny(name, "=:"); i >= 0 {
				var err error
				name, valstring = name[:i], name[i+1:]
				val, err = strconv.Atoi(valstring)
				if err != nil {
					val, haveInt = 1, false
				}
			}
			for _, t := range debugtab {
				if t.name != name {
					continue
				}
				switch vp := t.val.(type) {
				case nil:
					// Ignore
				case *string:
					*vp = valstring
				case *int:
					if !haveInt {
						log.Fatalf("invalid debug value %v", name)
					}
					*vp = val
				default:
					panic("bad debugtab type")
				}
				continue Split
			}
			// special case for ssa for now
			if strings.HasPrefix(name, "ssa/") {
				// expect form ssa/phase/flag
				// e.g. -d=ssa/generic_cse/time
				// _ in phase name also matches space
				phase := name[4:]
				flag := "debug" // default flag is debug
				if i := strings.Index(phase, "/"); i >= 0 {
					flag = phase[i+1:]
					phase = phase[:i]
				}
				err := ssa.PhaseOption(phase, flag, val, valstring)
				if err != "" {
					log.Fatalf(err)
				}
				continue Split
			}
			log.Fatalf("unknown debug key -d %s\n", name)
		}
	}

	if Flag.CompilingRuntime {
		// Runtime can't use -d=checkptr, at least not yet.
		Debug_checkptr = 0

		// Fuzzing the runtime isn't interesting either.
		Debug_libfuzzer = 0
	}

	// set via a -d flag
	Ctxt.Debugpcln = Debug_pctab
	if Flag.Dwarf {
		dwarf.EnableLogging(Debug_gendwarfinl != 0)
	}

	if Debug_softfloat != 0 {
		thearch.SoftFloat = true
	}

	// enable inlining.  for now:
	//	default: inlining on.  (Debug.l == 1)
	//	-l: inlining off  (Debug.l == 0)
	//	-l=2, -l=3: inlining on again, with extra debugging (Debug.l > 1)
	if Flag.LowerL <= 1 {
		Flag.LowerL = 1 - Flag.LowerL
	}

	if Flag.JSON != "" { // parse version,destination from json logging optimization.
		logopt.LogJsonOption(Flag.JSON)
	}
}

// concurrentFlagOk reports whether the current compiler flags
// are compatible with concurrent compilation.
func concurrentFlagOk() bool {
	// TODO(rsc): Many of these are fine. Remove them.
	return Flag.Percent == 0 &&
		Flag.E == 0 &&
		Flag.K == 0 &&
		Flag.L == 0 &&
		Flag.LowerH == 0 &&
		Flag.LowerJ == 0 &&
		Flag.LowerM == 0 &&
		Flag.LowerR == 0
}

func concurrentBackendAllowed() bool {
	if !concurrentFlagOk() {
		return false
	}

	// Debug.S by itself is ok, because all printing occurs
	// while writing the object file, and that is non-concurrent.
	// Adding Debug_vlog, however, causes Debug.S to also print
	// while flushing the plist, which happens concurrently.
	if Ctxt.Debugvlog || Flag.LowerD != "" || Flag.Live > 0 {
		return false
	}
	// TODO: Test and delete this condition.
	if objabi.Fieldtrack_enabled != 0 {
		return false
	}
	// TODO: fix races and enable the following flags
	if Ctxt.Flag_shared || Ctxt.Flag_dynlink || Flag.Race {
		return false
	}
	return true
}

func addImportDir(dir string) {
	if dir != "" {
		Flag.Cfg.ImportDirs = append(Flag.Cfg.ImportDirs, dir)
	}
}

func addImportMap(s string) {
	if Flag.Cfg.ImportMap == nil {
		Flag.Cfg.ImportMap = make(map[string]string)
	}
	if strings.Count(s, "=") != 1 {
		log.Fatal("-importmap argument must be of the form source=actual")
	}
	i := strings.Index(s, "=")
	source, actual := s[:i], s[i+1:]
	if source == "" || actual == "" {
		log.Fatal("-importmap argument must be of the form source=actual; source and actual must be non-empty")
	}
	Flag.Cfg.ImportMap[source] = actual
}

func readImportCfg(file string) {
	if Flag.Cfg.ImportMap == nil {
		Flag.Cfg.ImportMap = make(map[string]string)
	}
	Flag.Cfg.PackageFile = map[string]string{}
	data, err := ioutil.ReadFile(file)
	if err != nil {
		log.Fatalf("-importcfg: %v", err)
	}

	for lineNum, line := range strings.Split(string(data), "\n") {
		lineNum++ // 1-based
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		var verb, args string
		if i := strings.Index(line, " "); i < 0 {
			verb = line
		} else {
			verb, args = line[:i], strings.TrimSpace(line[i+1:])
		}
		var before, after string
		if i := strings.Index(args, "="); i >= 0 {
			before, after = args[:i], args[i+1:]
		}
		switch verb {
		default:
			log.Fatalf("%s:%d: unknown directive %q", file, lineNum, verb)
		case "importmap":
			if before == "" || after == "" {
				log.Fatalf(`%s:%d: invalid importmap: syntax is "importmap old=new"`, file, lineNum)
			}
			Flag.Cfg.ImportMap[before] = after
		case "packagefile":
			if before == "" || after == "" {
				log.Fatalf(`%s:%d: invalid packagefile: syntax is "packagefile path=filename"`, file, lineNum)
			}
			Flag.Cfg.PackageFile[before] = after
		}
	}
}

func readEmbedCfg(file string) {
	data, err := ioutil.ReadFile(file)
	if err != nil {
		log.Fatalf("-embedcfg: %v", err)
	}
	if err := json.Unmarshal(data, &Flag.Cfg.Embed); err != nil {
		log.Fatalf("%s: %v", file, err)
	}
	if Flag.Cfg.Embed.Patterns == nil {
		log.Fatalf("%s: invalid embedcfg: missing Patterns", file)
	}
	if Flag.Cfg.Embed.Files == nil {
		log.Fatalf("%s: invalid embedcfg: missing Files", file)
	}
}
