// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package base

import (
	"encoding/json"
	"flag"
	"fmt"
	"internal/buildcfg"
	"io/ioutil"
	"log"
	"os"
	"reflect"
	"runtime"
	"strings"

	"cmd/internal/objabi"
	"cmd/internal/sys"
)

func usage() {
	fmt.Fprintf(os.Stderr, "usage: compile [options] file.go...\n")
	objabi.Flagprint(os.Stderr)
	Exit(2)
}

// Flag holds the parsed command-line flags.
// See ParseFlag for non-zero defaults.
var Flag CmdFlags

// A CountFlag is a counting integer flag.
// It accepts -name=value to set the value directly,
// but it also accepts -name with no =value to increment the count.
type CountFlag int

// CmdFlags defines the command-line flags (see var Flag).
// Each struct field is a different flag, by default named for the lower-case of the field name.
// If the flag name is a single letter, the default flag name is left upper-case.
// If the flag name is "Lower" followed by a single letter, the default flag name is the lower-case of the last letter.
//
// If this default flag name can't be made right, the `flag` struct tag can be used to replace it,
// but this should be done only in exceptional circumstances: it helps everyone if the flag name
// is obvious from the field name when the flag is used elsewhere in the compiler sources.
// The `flag:"-"` struct tag makes a field invisible to the flag logic and should also be used sparingly.
//
// Each field must have a `help` struct tag giving the flag help message.
//
// The allowed field types are bool, int, string, pointers to those (for values stored elsewhere),
// CountFlag (for a counting flag), and func(string) (for a flag that uses special code for parsing).
type CmdFlags struct {
	// Single letters
	B CountFlag    "help:\"disable bounds checking\""
	C CountFlag    "help:\"disable printing of columns in error messages\""
	D string       "help:\"set relative `path` for local imports\""
	E CountFlag    "help:\"debug symbol export\""
	G CountFlag    "help:\"accept generic code\""
	I func(string) "help:\"add `directory` to import search path\""
	K CountFlag    "help:\"debug missing line numbers\""
	L CountFlag    "help:\"show full file names in error messages\""
	N CountFlag    "help:\"disable optimizations\""
	S CountFlag    "help:\"print assembly listing\""
	// V is added by objabi.AddVersionFlag
	W CountFlag "help:\"debug parse tree after type checking\""

	LowerC int        "help:\"concurrency during compilation (1 means no concurrency)\""
	LowerD flag.Value "help:\"enable debugging settings; try -d help\""
	LowerE CountFlag  "help:\"no limit on number of errors reported\""
	LowerH CountFlag  "help:\"halt on error\""
	LowerJ CountFlag  "help:\"debug runtime-initialized variables\""
	LowerL CountFlag  "help:\"disable inlining\""
	LowerM CountFlag  "help:\"print optimization decisions\""
	LowerO string     "help:\"write output to `file`\""
	LowerP *string    "help:\"set expected package import `path`\"" // &Ctxt.Pkgpath, set below
	LowerR CountFlag  "help:\"debug generated wrappers\""
	LowerT bool       "help:\"enable tracing for debugging the compiler\""
	LowerW CountFlag  "help:\"debug type checking\""
	LowerV *bool      "help:\"increase debug verbosity\""

	// Special characters
	Percent          int  "flag:\"%\" help:\"debug non-static initializers\""
	CompilingRuntime bool "flag:\"+\" help:\"compiling runtime\""

	// Longer names
	AsmHdr             string       "help:\"write assembly header to `file`\""
	ASan               bool         "help:\"build code compatible with C/C++ address sanitizer\""
	Bench              string       "help:\"append benchmark times to `file`\""
	BlockProfile       string       "help:\"write block profile to `file`\""
	BuildID            string       "help:\"record `id` as the build id in the export metadata\""
	CPUProfile         string       "help:\"write cpu profile to `file`\""
	Complete           bool         "help:\"compiling complete package (no C or assembly)\""
	ClobberDead        bool         "help:\"clobber dead stack slots (for debugging)\""
	ClobberDeadReg     bool         "help:\"clobber dead registers (for debugging)\""
	Dwarf              bool         "help:\"generate DWARF symbols\""
	DwarfBASEntries    *bool        "help:\"use base address selection entries in DWARF\""                        // &Ctxt.UseBASEntries, set below
	DwarfLocationLists *bool        "help:\"add location lists to DWARF in optimized mode\""                      // &Ctxt.Flag_locationlists, set below
	Dynlink            *bool        "help:\"support references to Go symbols defined in other shared libraries\"" // &Ctxt.Flag_dynlink, set below
	EmbedCfg           func(string) "help:\"read go:embed configuration from `file`\""
	GenDwarfInl        int          "help:\"generate DWARF inline info records\"" // 0=disabled, 1=funcs, 2=funcs+formals/locals
	GoVersion          string       "help:\"required version of the runtime\""
	ImportCfg          func(string) "help:\"read import configuration from `file`\""
	ImportMap          func(string) "help:\"add `definition` of the form source=actual to import map\""
	InstallSuffix      string       "help:\"set pkg directory `suffix`\""
	JSON               string       "help:\"version,file for JSON compiler/optimizer detail output\""
	Lang               string       "help:\"Go language version source code expects\""
	LinkObj            string       "help:\"write linker-specific object to `file`\""
	LinkShared         *bool        "help:\"generate code that will be linked against Go shared libraries\"" // &Ctxt.Flag_linkshared, set below
	Live               CountFlag    "help:\"debug liveness analysis\""
	MSan               bool         "help:\"build code compatible with C/C++ memory sanitizer\""
	MemProfile         string       "help:\"write memory profile to `file`\""
	MemProfileRate     int          "help:\"set runtime.MemProfileRate to `rate`\""
	MutexProfile       string       "help:\"write mutex profile to `file`\""
	NoLocalImports     bool         "help:\"reject local (relative) imports\""
	Pack               bool         "help:\"write to file.a instead of file.o\""
	Race               bool         "help:\"enable race detector\""
	Shared             *bool        "help:\"generate code that can be linked into a shared library\"" // &Ctxt.Flag_shared, set below
	SmallFrames        bool         "help:\"reduce the size limit for stack allocated objects\""      // small stacks, to diagnose GC latency; see golang.org/issue/27732
	Spectre            string       "help:\"enable spectre mitigations in `list` (all, index, ret)\""
	Std                bool         "help:\"compiling standard library\""
	SymABIs            string       "help:\"read symbol ABIs from `file`\""
	TraceProfile       string       "help:\"write an execution trace to `file`\""
	TrimPath           string       "help:\"remove `prefix` from recorded source file paths\""
	WB                 bool         "help:\"enable write barrier\"" // TODO: remove

	// Configuration derived from flags; not a flag itself.
	Cfg struct {
		Embed struct { // set by -embedcfg
			Patterns map[string][]string
			Files    map[string]string
		}
		ImportDirs   []string          // appended to by -I
		ImportMap    map[string]string // set by -importmap OR -importcfg
		PackageFile  map[string]string // set by -importcfg; nil means not in use
		SpectreIndex bool              // set by -spectre=index or -spectre=all
		// Whether we are adding any sort of code instrumentation, such as
		// when the race detector is enabled.
		Instrumenting bool
	}
}

// ParseFlags parses the command-line flags into Flag.
func ParseFlags() {
	Flag.G = 3
	Flag.I = addImportDir

	Flag.LowerC = 1
	Flag.LowerD = objabi.NewDebugFlag(&Debug, DebugSSA)
	Flag.LowerP = &Ctxt.Pkgpath
	Flag.LowerV = &Ctxt.Debugvlog

	Flag.Dwarf = buildcfg.GOARCH != "wasm"
	Flag.DwarfBASEntries = &Ctxt.UseBASEntries
	Flag.DwarfLocationLists = &Ctxt.Flag_locationlists
	*Flag.DwarfLocationLists = true
	Flag.Dynlink = &Ctxt.Flag_dynlink
	Flag.EmbedCfg = readEmbedCfg
	Flag.GenDwarfInl = 2
	Flag.ImportCfg = readImportCfg
	Flag.ImportMap = addImportMap
	Flag.LinkShared = &Ctxt.Flag_linkshared
	Flag.Shared = &Ctxt.Flag_shared
	Flag.WB = true

	Debug.InlFuncsWithClosures = 1
	if buildcfg.Experiment.Unified {
		Debug.Unified = 1
	}

	Debug.Checkptr = -1 // so we can tell whether it is set explicitly

	Flag.Cfg.ImportMap = make(map[string]string)

	objabi.AddVersionFlag() // -V
	registerFlags()
	objabi.Flagparse(usage)

	if Flag.MSan && !sys.MSanSupported(buildcfg.GOOS, buildcfg.GOARCH) {
		log.Fatalf("%s/%s does not support -msan", buildcfg.GOOS, buildcfg.GOARCH)
	}
	if Flag.ASan && !sys.ASanSupported(buildcfg.GOOS, buildcfg.GOARCH) {
		log.Fatalf("%s/%s does not support -asan", buildcfg.GOOS, buildcfg.GOARCH)
	}
	if Flag.Race && !sys.RaceDetectorSupported(buildcfg.GOOS, buildcfg.GOARCH) {
		log.Fatalf("%s/%s does not support -race", buildcfg.GOOS, buildcfg.GOARCH)
	}
	if (*Flag.Shared || *Flag.Dynlink || *Flag.LinkShared) && !Ctxt.Arch.InFamily(sys.AMD64, sys.ARM, sys.ARM64, sys.I386, sys.PPC64, sys.RISCV64, sys.S390X) {
		log.Fatalf("%s/%s does not support -shared", buildcfg.GOOS, buildcfg.GOARCH)
	}
	parseSpectre(Flag.Spectre) // left as string for RecordFlags

	Ctxt.Flag_shared = Ctxt.Flag_dynlink || Ctxt.Flag_shared
	Ctxt.Flag_optimize = Flag.N == 0
	Ctxt.Debugasm = int(Flag.S)
	Ctxt.Flag_maymorestack = Debug.MayMoreStack

	if flag.NArg() < 1 {
		usage()
	}

	if Flag.GoVersion != "" && Flag.GoVersion != runtime.Version() {
		fmt.Printf("compile: version %q does not match go tool version %q\n", runtime.Version(), Flag.GoVersion)
		Exit(2)
	}

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
	switch {
	case Flag.Race && Flag.MSan:
		log.Fatal("cannot use both -race and -msan")
	case Flag.Race && Flag.ASan:
		log.Fatal("cannot use both -race and -asan")
	case Flag.MSan && Flag.ASan:
		log.Fatal("cannot use both -msan and -asan")
	}
	if Flag.Race || Flag.MSan || Flag.ASan {
		// -race, -msan and -asan imply -d=checkptr for now.
		if Debug.Checkptr == -1 { // if not set explicitly
			Debug.Checkptr = 1
		}
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

	if Flag.CompilingRuntime {
		// Runtime can't use -d=checkptr, at least not yet.
		Debug.Checkptr = 0

		// Fuzzing the runtime isn't interesting either.
		Debug.Libfuzzer = 0
	}

	if Debug.Checkptr == -1 { // if not set explicitly
		Debug.Checkptr = 0
	}

	// set via a -d flag
	Ctxt.Debugpcln = Debug.PCTab
}

// registerFlags adds flag registrations for all the fields in Flag.
// See the comment on type CmdFlags for the rules.
func registerFlags() {
	var (
		boolType      = reflect.TypeOf(bool(false))
		intType       = reflect.TypeOf(int(0))
		stringType    = reflect.TypeOf(string(""))
		ptrBoolType   = reflect.TypeOf(new(bool))
		ptrIntType    = reflect.TypeOf(new(int))
		ptrStringType = reflect.TypeOf(new(string))
		countType     = reflect.TypeOf(CountFlag(0))
		funcType      = reflect.TypeOf((func(string))(nil))
	)

	v := reflect.ValueOf(&Flag).Elem()
	t := v.Type()
	for i := 0; i < t.NumField(); i++ {
		f := t.Field(i)
		if f.Name == "Cfg" {
			continue
		}

		var name string
		if len(f.Name) == 1 {
			name = f.Name
		} else if len(f.Name) == 6 && f.Name[:5] == "Lower" && 'A' <= f.Name[5] && f.Name[5] <= 'Z' {
			name = string(rune(f.Name[5] + 'a' - 'A'))
		} else {
			name = strings.ToLower(f.Name)
		}
		if tag := f.Tag.Get("flag"); tag != "" {
			name = tag
		}

		help := f.Tag.Get("help")
		if help == "" {
			panic(fmt.Sprintf("base.Flag.%s is missing help text", f.Name))
		}

		if k := f.Type.Kind(); (k == reflect.Ptr || k == reflect.Func) && v.Field(i).IsNil() {
			panic(fmt.Sprintf("base.Flag.%s is uninitialized %v", f.Name, f.Type))
		}

		switch f.Type {
		case boolType:
			p := v.Field(i).Addr().Interface().(*bool)
			flag.BoolVar(p, name, *p, help)
		case intType:
			p := v.Field(i).Addr().Interface().(*int)
			flag.IntVar(p, name, *p, help)
		case stringType:
			p := v.Field(i).Addr().Interface().(*string)
			flag.StringVar(p, name, *p, help)
		case ptrBoolType:
			p := v.Field(i).Interface().(*bool)
			flag.BoolVar(p, name, *p, help)
		case ptrIntType:
			p := v.Field(i).Interface().(*int)
			flag.IntVar(p, name, *p, help)
		case ptrStringType:
			p := v.Field(i).Interface().(*string)
			flag.StringVar(p, name, *p, help)
		case countType:
			p := (*int)(v.Field(i).Addr().Interface().(*CountFlag))
			objabi.Flagcount(name, help, p)
		case funcType:
			f := v.Field(i).Interface().(func(string))
			objabi.Flagfn1(name, help, f)
		default:
			if val, ok := v.Field(i).Interface().(flag.Value); ok {
				flag.Var(val, name, help)
			} else {
				panic(fmt.Sprintf("base.Flag.%s has unexpected type %s", f.Name, f.Type))
			}
		}
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
	if Ctxt.Debugvlog || Debug.Any || Flag.Live > 0 {
		return false
	}
	// TODO: Test and delete this condition.
	if buildcfg.Experiment.FieldTrack {
		return false
	}
	// TODO: fix races and enable the following flags
	if Ctxt.Flag_dynlink || Flag.Race {
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

// parseSpectre parses the spectre configuration from the string s.
func parseSpectre(s string) {
	for _, f := range strings.Split(s, ",") {
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
		switch buildcfg.GOARCH {
		case "amd64":
			// ok
		default:
			log.Fatalf("GOARCH=%s does not support -spectre=index", buildcfg.GOARCH)
		}
	}
}
