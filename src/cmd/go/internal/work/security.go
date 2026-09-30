// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Checking of compiler and linker flags.
// We must avoid flags like -fplugin=, which can allow
// arbitrary code execution during the build.
// Do not make changes here without carefully
// considering the implications.
// (That's why the code is isolated in a file named security.go.)
//
// Note that -Wl,foo means split foo on commas and pass to
// the linker, so that -Wl,-foo,bar means pass -foo bar to
// the linker. Similarly -Wa,foo for the assembler and so on.
// If any of these are permitted, the wildcard portion must
// disallow commas.
//
// Note also that GNU binutils accept any argument @foo
// as meaning "read more flags from the file foo", so we must
// guard against any command-line argument beginning with @,
// even things like "-I @foo".
// We use load.SafeArg (which is even more conservative)
// to reject these.
//
// Even worse, gcc -I@foo (one arg) turns into cc1 -I @foo (two args),
// so although gcc doesn't expand the @foo, cc1 will.
// So out of paranoia, we reject @ at the beginning of every
// flag argument that might be split into its own argument.

package work

import (
	"bytes"
	"debug/elf"
	"debug/macho"
	"debug/pe"
	"debug/plan9obj"
	"encoding/binary"
	"fmt"
	"internal/lazyregexp"
	"internal/xcoff"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/load"
	"cmd/go/internal/str"
)

var re = lazyregexp.New

var validCompilerFlags = []*lazyregexp.Regexp{
	re(`-D([A-Za-z_][A-Za-z0-9_]*)(=[^@\-]*)?`),
	re(`-U([A-Za-z_][A-Za-z0-9_]*)`),
	re(`-F([^@\-].*)`),
	re(`-I([^@\-].*)`),
	re(`-O`),
	re(`-O([^@\-].*)`),
	re(`-W`),
	re(`-W([^@,]+)`), // -Wall but not -Wa,-foo.
	re(`-Wa,-mbig-obj`),
	re(`-Wp,-D([A-Za-z_][A-Za-z0-9_]*)(=[^@,\-]*)?`),
	re(`-Wp,-U([A-Za-z_][A-Za-z0-9_]*)`),
	re(`-ansi`),
	re(`-f(no-)?asynchronous-unwind-tables`),
	re(`-f(no-)?blocks`),
	re(`-f(no-)builtin-[a-zA-Z0-9_]*`),
	re(`-f(no-)?common`),
	re(`-f(no-)?constant-cfstrings`),
	re(`-fdebug-prefix-map=([^@]+)=([^@]+)`),
	re(`-fdiagnostics-show-note-include-stack`),
	re(`-ffile-prefix-map=([^@]+)=([^@]+)`),
	re(`-fno-canonical-system-headers`),
	re(`-f(no-)?eliminate-unused-debug-types`),
	re(`-f(no-)?exceptions`),
	re(`-f(no-)?fast-math`),
	re(`-f(no-)?inline-functions`),
	re(`-finput-charset=([^@\-].*)`),
	re(`-f(no-)?fat-lto-objects`),
	re(`-f(no-)?keep-inline-dllexport`),
	re(`-f(no-)?lto`),
	re(`-fmacro-backtrace-limit=(.+)`),
	re(`-fmessage-length=(.+)`),
	re(`-f(no-)?modules`),
	re(`-f(no-)?objc-arc`),
	re(`-f(no-)?objc-nonfragile-abi`),
	re(`-f(no-)?objc-legacy-dispatch`),
	re(`-f(no-)?omit-frame-pointer`),
	re(`-f(no-)?openmp(-simd)?`),
	re(`-f(no-)?permissive`),
	re(`-f(no-)?(pic|PIC|pie|PIE)`),
	re(`-f(no-)?plt`),
	re(`-f(no-)?rtti`),
	re(`-f(no-)?split-stack`),
	re(`-f(no-)?stack-(.+)`),
	re(`-f(no-)?strict-aliasing`),
	re(`-f(un)signed-char`),
	re(`-f(no-)?use-linker-plugin`), // safe if -B is not used; we don't permit -B
	re(`-f(no-)?visibility-inlines-hidden`),
	re(`-fsanitize=([^@\-].*)`),
	re(`-fsanitize-undefined-strip-path-components=(-)?[0-9]+`),
	re(`-ftemplate-depth-(.+)`),
	re(`-ftls-model=(global-dynamic|local-dynamic|initial-exec|local-exec)`),
	re(`-fvisibility=(.+)`),
	re(`-g([^@\-].*)?`),
	re(`-m32`),
	re(`-m64`),
	re(`-m(abi|arch|cpu|fpu|simd|tls-dialect|tune)=([^@\-].*)`),
	re(`-m(no-)?v?aes`),
	re(`-marm`),
	re(`-mcmodel=[0-9a-z-]+`),
	re(`-mfloat-abi=([^@\-].*)`),
	re(`-m(soft|single|double)-float`),
	re(`-mfpmath=[0-9a-z,+]*`),
	re(`-m(no-)?avx[0-9a-z.]*`),
	re(`-m(no-)?ms-bitfields`),
	re(`-m(no-)?stack-(.+)`),
	re(`-mmacosx-(.+)`),
	re(`-m(no-)?relax`),
	re(`-m(no-)?strict-align`),
	re(`-m(no-)?(lsx|lasx|frecipe|div32|lam-bh|lamcas|ld-seq-sa)`),
	re(`-mios-simulator-version-min=(.+)`),
	re(`-miphoneos-version-min=(.+)`),
	re(`-mlarge-data-threshold=[0-9]+`),
	re(`-mtvos-simulator-version-min=(.+)`),
	re(`-mtvos-version-min=(.+)`),
	re(`-mwatchos-simulator-version-min=(.+)`),
	re(`-mwatchos-version-min=(.+)`),
	re(`-mnop-fun-dllimport`),
	re(`-m(no-)?sse[0-9.]*`),
	re(`-m(no-)?ssse3`),
	re(`-mthumb(-interwork)?`),
	re(`-mthreads`),
	re(`-mwindows`),
	re(`-no-canonical-prefixes`),
	re(`--param=ssp-buffer-size=[0-9]*`),
	re(`-pedantic(-errors)?`),
	re(`-pipe`),
	re(`-pthread`),
	re(`--static`),
	re(`-?-std=([^@\-].*)`),
	re(`-?-stdlib=([^@\-].*)`),
	re(`-w`),
	re(`-x(c|c-header|cpp-output|c\+\+|c\+\+-header|c\+\+-cpp-output|objective-c|objective-c-header|objective-c-cpp-output|objective-c\+\+|objective-c\+\+-header|objective-c\+\+-cpp-output|assembler|assembler-with-cpp)`),
	re(`-v`),
}

var validCompilerXLang = re(`^(c|c-header|cpp-output|c\+\+|c\+\+-header|c\+\+-cpp-output|objective-c|objective-c-header|objective-c-cpp-output|objective-c\+\+|objective-c\+\+-header|objective-c\+\+-cpp-output|assembler|assembler-with-cpp)$`)

var validCompilerFlagsWithNextArg = []string{
	"-arch",
	"-D",
	"-U",
	"-I",
	"-F",
	"-framework",
	"-include",
	"-isysroot",
	"-isystem",
	"-x",
}

var invalidCompilerFlags = []*lazyregexp.Regexp{
	// Clang's -gen-cdb-fragment-path takes a directory argument and writes
	// compilation database fragments there; have to exclude separately because
	// -g* is allowed in general. GCC's -gnatO similarly takes a separate
	// output filename argument in the GCC driver.
	re(`-gen-.*`),
	re(`-gnat.*`),
}

var invalidLinkerFlags = []*lazyregexp.Regexp{
	// On macOS this means the linker loads and executes the next argument.
	// Have to exclude separately because -lfoo is allowed in general.
	re(`-lto_library`),
	re(`-lazy_library`),
	re(`-lazy_framework`),
	re(`-load_hidden`),
	// GNU gold and LLD accept single-dash -library and -library-path.
	re(`-library(-path)?(=.*)?`),
	re(`-gen-.*`),
	re(`-gnat.*`),
	// In lld, --warn-backrefs-exclude takes a separate <glob> argument when not
	// joined with '='; exclude the bare form because -Wl,--(no-)?warn-([^,]+) is
	// allowed in general.
	re(`-Wl,--warn-backrefs-exclude`),
}

var validLinkerFlags = []*lazyregexp.Regexp{
	re(`-F([^@\-].*)`),
	re(`-l([^@\-:=/\\][^:=/\\]*)`),
	re(`-L([^@\-].*)`),
	re(`-O`),
	re(`-O([^@\-].*)`),
	re(`-f(no-)?(pic|PIC|pie|PIE)`),
	re(`-f(no-)?openmp(-simd)?`),
	re(`-fsanitize=([^@\-].*)`),
	re(`-flat_namespace`),
	re(`-g([^@\-].*)?`),
	re(`-headerpad_max_install_names`),
	re(`-m(abi|arch|cpu|fpu|simd|tls-dialect|tune)=([^@\-].*)`),
	re(`-mcmodel=[0-9a-z-]+`),
	re(`-mfloat-abi=([^@\-].*)`),
	re(`-m(soft|single|double)-float`),
	re(`-m(no-)?relax`),
	re(`-m(no-)?strict-align`),
	re(`-m(no-)?(lsx|lasx|frecipe|div32|lam-bh|lamcas|ld-seq-sa)`),
	re(`-mmacosx-(.+)`),
	re(`-mios-simulator-version-min=(.+)`),
	re(`-miphoneos-version-min=(.+)`),
	re(`-mthreads`),
	re(`-mwindows`),
	re(`-(pic|PIC|pie|PIE)`),
	re(`-pthread`),
	re(`-rdynamic`),
	re(`-shared`),
	re(`-?-static([-a-z0-9+]*)`),
	re(`-?-stdlib=([^@\-].*)`),
	re(`-v`),

	// Note that any wildcards in -Wl need to exclude comma,
	// since -Wl splits its argument at commas and passes
	// them all to the linker uninterpreted. Allowing comma
	// in a wildcard would allow tunneling arbitrary additional
	// linker arguments through one of these.
	re(`-Wl,--(no-)?allow-multiple-definition`),
	re(`-Wl,--(no-)?allow-shlib-undefined`),
	re(`-Wl,--(no-)?as-needed`),
	re(`-Wl,-Bdynamic`),
	re(`-Wl,-berok`),
	re(`-Wl,-Bstatic`),
	re(`-Wl,-Bsymbolic-functions`),
	re(`-Wl,-O[0-9]+`),
	re(`-Wl,-d[ny]`),
	re(`-Wl,--disable-new-dtags`),
	re(`-Wl,-e[=,][a-zA-Z0-9]+`),
	re(`-Wl,--enable-new-dtags`),
	re(`-Wl,--end-group`),
	re(`-Wl,--(no-)?export-dynamic`),
	re(`-Wl,-E`),
	re(`-Wl,-framework,[^,@\-][^,]*`),
	re(`-Wl,--hash-style=(sysv|gnu|both)`),
	re(`-Wl,-headerpad_max_install_names`),
	re(`-Wl,--no-undefined`),
	re(`-Wl,--pop-state`),
	re(`-Wl,--push-state`),
	re(`-Wl,-R[,=]?([^@\-,=][^,@]*$)`),
	re(`-Wl,--just-symbols[=,]([^,@\-][^,@]*)`),
	re(`-Wl,-rpath(-link)?[=,]([^,@\-][^,]*)`),
	re(`-Wl,-s`),
	re(`-Wl,-search_paths_first`),
	re(`-Wl,-sectcreate,([^,@\-][^,]*),([^,@\-][^,]*),([^,@\-][^,]*)`),
	re(`-Wl,--start-group`),
	re(`-Wl,-?-static`),
	re(`-Wl,-?-subsystem,(native|windows|console|posix|xbox)`),
	re(`-Wl,-syslibroot[=,]([^,@\-][^,]*)`),
	re(`-Wl,-undefined[=,]([^,@\-][^,]*)`),
	re(`-Wl,-?-unresolved-symbols=[^,]+`),
	re(`-Wl,--(no-)?warn-([^,]+)`),
	re(`-Wl,-?-wrap[=,][^,@\-][^,]*`),
	re(`-Wl(,-z,(relro|now|(no)?execstack))+`),

	re(`[a-zA-Z0-9_/].*\.(a|o|obj|dll|dylib|lib|so|tbd)`), // direct linker inputs: x.o or libfoo.so (but not -foo.o or @foo.o)
	re(`\./.*\.(a|o|obj|dll|dylib|lib|so|tbd)`),
}

var validLinkerFlagsWithNextArg = []string{
	"-arch",
	"-F",
	"-l",
	"-L",
	"-framework",
	"-isysroot",
	"-Wl,-framework",
	"-Wl,-rpath",
	"-Wl,-R",
	"-Wl,--just-symbols",
	"-Wl,-undefined",
}

// recombinedLinkerFlagsWithNextArg lists flags in validLinkerFlagsWithNextArg
// that GCC and Clang recombine with their following argument (e.g., "-l" "foo"
// becomes "-lfoo"), which must also be checked against invalidLinkerFlags.
var recombinedLinkerFlagsWithNextArg = []string{
	"-l",
}

var validPkgConfigFlags = []*lazyregexp.Regexp{
	re(`--atleast-pkgconfig-version=\d+\.\d+\.\d+`),
	re(`--atleast-version=\d+\.\d+\.\d+`),
	re(`--cflags-only-I`),
	re(`--cflags`),
	re(`--define-prefix`),
	re(`--define-variable=[A-Za-z_][A-Za-z0-9_]*=[^@\-]*`),
	re(`--digraph`),
	re(`--dont-define-prefix`),
	re(`--dont-relocate-paths`),
	re(`--dump-personality`),
	re(`--env-only`),
	re(`--errors-to-stdout`),
	re(`--exact-version=\d+\.\d+\.\d+`),
	re(`--exists`),
	re(`--fragment-filter=[A-Za-z_][a-zA-Z0-9_]*`),
	re(`--ignore-conflicts`),
	re(`--internal-cflags`),
	re(`--keep-system-cflags`),
	re(`--keep-system-libs`),
	re(`--libs-only-l`),
	re(`--libs-only-L`),
	re(`--libs`),
	re(`--list-all`),
	re(`--list-package-names`),
	re(`--max-version=\d+\.\d+\.\d+`),
	re(`--maximum-traverse-depth=[0-9]+`),
	re(`--modversion`),
	re(`--msvc-syntax`),
	re(`--no-cache`),
	re(`--no-provides`),
	re(`--no-uninstalled`),
	re(`--path`),
	re(`--personality=(triplet|filename)`),
	re(`--prefix-variable=[A-Za-z_][a-zA-Z0-9_]*`),
	re(`--print-errors`),
	re(`--print-provides`),
	re(`--print-requires-private`),
	re(`--print-requires`),
	re(`--print-variables`),
	re(`--pure`),
	re(`--shared`),
	re(`--short-errors`),
	re(`--silence-errors`),
	re(`--simulate`),
	re(`--static`),
	re(`--uninstalled`),
	re(`--validate`),
	re(`--variable=[A-Za-z_][a-zA-Z0-9_]*`),
	re(`--with-path=[^@\-].*`),
}

// hasGCCSpecFlags reports whether list contains linker flags that cause GCC's
// built-in *link_command spec to evaluate %:include(libgomp.spec) (-fopenmp or
// -fopenacc) or %:include(libsanitizer.spec) (-fsanitize=* combined with
// -static or -static-lib*).
func hasGCCSpecFlags(list []string) bool {
	hasSanitize := false
	hasStatic := false
	for _, arg := range list {
		if arg == "-fopenmp" || arg == "-fopenacc" {
			return true
		}
		if strings.HasPrefix(arg, "-fsanitize=") {
			hasSanitize = true
		}
		if arg == "-static" || arg == "--static" || strings.HasPrefix(arg, "-static-") || strings.HasPrefix(arg, "--static-") {
			hasStatic = true
		}
	}
	return hasSanitize && hasStatic
}

func checkCompilerFlags(name, source string, list []string) error {
	checkOverrides := true
	return checkFlags(name, source, list, invalidCompilerFlags, validCompilerFlags, nil, validCompilerFlagsWithNextArg, checkOverrides)
}

func checkLinkerFlags(name, source string, list []string) error {
	checkOverrides := true
	if err := checkFlags(name, source, list, invalidLinkerFlags, validLinkerFlags, recombinedLinkerFlagsWithNextArg, validLinkerFlagsWithNextArg, checkOverrides); err != nil {
		return err
	}
	if hasGCCSpecFlags(list) {
		if err := checkLinkerSpecDir(source, "."); err != nil {
			return err
		}
	}
	for i := 0; i < len(list); i++ {
		arg := list[i]
		switch {
		case strings.HasPrefix(arg, "-L"):
			dir := arg[2:]
			if dir == "" && i+1 < len(list) {
				i++
				dir = list[i]
			}
			if dir != "" {
				if err := checkLinkerLibDir(source, dir); err != nil {
					return err
				}
			}
		// Although -isysroot is a preprocessor/compiler option in GCC, clang on
		// Darwin also accepts it as a linker flag (forwarding it as -syslibroot)
		// and we include it in validLinkerFlagsWithNextArg on all platforms.
		case arg == "-isysroot" && i+1 < len(list):
			i++ // consume next argument
			if err := checkLinkerSysroot(source, arg+" "+list[i], list[i]); err != nil {
				return err
			}
		case arg == "-framework" && i+1 < len(list):
			i++
			if err := checkLinkerNonPathNonFile(source, arg+" "+list[i], list[i]); err != nil {
				return err
			}
		case strings.HasPrefix(arg, "-Wl,"):
			raw := arg
			warg := arg
			if slices.Contains(validLinkerFlagsWithNextArg, arg) && i+1 < len(list) {
				i++
				raw = arg + " " + list[i]
				warg = arg + "," + strings.TrimPrefix(list[i], "-Wl,")
			}
			if err := checkLinkerWlFlag(source, raw, warg); err != nil {
				return err
			}
		case reDirectLinkerInput.MatchString(arg) && isLocalLinkerPath(arg):
			if err := checkLinkerInputFile(source, arg); err != nil {
				return err
			}
		}
	}
	return nil
}

// checkLinkerSysroot verifies that the target of -isysroot or -Wl,-syslibroot
// does not name a regular file on disk, which GNU ld.bfd would treat as a
// positional linker script.
func checkLinkerSysroot(source, flag, dir string) error {
	if fi, err := os.Stat(dir); err == nil && !fi.IsDir() {
		return fmt.Errorf("invalid flag in %s: %s (not a directory)", source, flag)
	}
	return checkLinkerSpecDir(source, dir)
}

// checkLinkerNonPathNonFile verifies that val (such as a framework or symbol
// name) is not a path and does not name a regular file in the current working
// directory. If a Darwin-only or symbol flag is misinterpreted by GNU ld.bfd
// (for example, -framework <val> parsed as -f ramework <val>, or -undefined
// <val> parsed as -u ndefined <val>), ld.bfd treats <val> as a positional
// input file and will execute it as a linker script if it is a regular file.
// Directories (such as "." in existing tests or subdirectories in the working
// directory that happen to share a framework or symbol name) are allowed
// because ld.bfd only parses regular files as linker scripts.
func checkLinkerNonPathNonFile(source, raw, val string) error {
	if strings.ContainsAny(val, `/\`) {
		return fmt.Errorf("invalid flag in %s: %s (see https://go.dev/s/invalidflag)", source, raw)
	}
	if fi, err := os.Stat(val); err == nil && !fi.IsDir() {
		return fmt.Errorf("invalid flag in %s: %s (see https://go.dev/s/invalidflag)", source, raw)
	}
	return nil
}

func checkLinkerObjOrDir(source, raw, val string) error {
	if !isLocalLinkerPath(val) {
		return nil
	}
	if fi, err := os.Stat(val); err == nil && !fi.IsDir() {
		if err := checkLinkerInputFile(source, val); err != nil {
			return fmt.Errorf("invalid flag in %s: %s (%v)", source, raw, err)
		}
	}
	return nil
}

func checkLinkerWlFlag(source, raw, arg string) error {
	wargs := []string{arg}
	if strings.HasPrefix(arg, "-Wl,--push-state,") {
		for _, sub := range strings.Split(arg, ",")[2:] {
			wargs = append(wargs, "-Wl,"+sub)
		}
	}
	for _, warg := range wargs {
		switch {
		case strings.HasPrefix(warg, "-Wl,-syslibroot,"), strings.HasPrefix(warg, "-Wl,-syslibroot="):
			dir := warg[len("-Wl,-syslibroot,"):]
			if err := checkLinkerSysroot(source, raw, dir); err != nil {
				return err
			}
		case strings.HasPrefix(warg, "-Wl,-framework,"):
			if err := checkLinkerNonPathNonFile(source, raw, warg[len("-Wl,-framework,"):]); err != nil {
				return err
			}
		case strings.HasPrefix(warg, "-Wl,-undefined,"), strings.HasPrefix(warg, "-Wl,-undefined="):
			if err := checkLinkerNonPathNonFile(source, raw, warg[len("-Wl,-undefined,"):]); err != nil {
				return err
			}
		case strings.HasPrefix(warg, "-Wl,--just-symbols,"), strings.HasPrefix(warg, "-Wl,--just-symbols="):
			if err := checkLinkerObjOrDir(source, raw, warg[len("-Wl,--just-symbols,"):]); err != nil {
				return err
			}
		case strings.HasPrefix(warg, "-Wl,-R"):
			val := warg[len("-Wl,-R"):]
			if after, ok := strings.CutPrefix(val, ","); ok {
				val = after
			} else if after, ok := strings.CutPrefix(val, "="); ok {
				val = after
			}
			if err := checkLinkerObjOrDir(source, raw, val); err != nil {
				return err
			}
		case strings.HasPrefix(warg, "-Wl,-subsystem,"):
			// On non-Windows targets, GNU ld.gold (and older non-PE ld.bfd) parses
			// single-dash -subsystem as -s -u bsystem and treats the following
			// comma-split argument as a positional input file / linker script.
			if cfg.Goos != "windows" {
				if err := checkLinkerNonPathNonFile(source, raw, warg[len("-Wl,-subsystem,"):]); err != nil {
					return err
				}
			}
		case strings.HasPrefix(warg, "-Wl,-sectcreate,"):
			// -sectcreate <seg> <sect> <file> is an Apple linker flag whose
			// third argument is legitimately a regular file on Darwin/iOS.
			// On other platforms, GNU ld.bfd parses -sectcreate as -s -e ctcreate
			// and treats all three arguments as positional linker scripts; we still
			// allow non-existent file arguments on non-Darwin so cross-platform
			// flag validation tests continue to work.
			parts := strings.SplitN(warg[len("-Wl,-sectcreate,"):], ",", 3)
			if len(parts) == 3 {
				if err := checkLinkerNonPathNonFile(source, raw, parts[0]); err != nil {
					return err
				}
				if err := checkLinkerNonPathNonFile(source, raw, parts[1]); err != nil {
					return err
				}
				if cfg.Goos != "darwin" && cfg.Goos != "ios" {
					if fi, err := os.Stat(parts[2]); err == nil && !fi.IsDir() {
						return fmt.Errorf("invalid flag in %s: %s (see https://go.dev/s/invalidflag)", source, raw)
					}
				}
			}
		}
	}
	return nil
}

func trimLinkerSysrootPrefix(dir string) string {
	if strings.HasPrefix(dir, "=") {
		return dir[1:]
	}
	if strings.HasPrefix(dir, "$SYSROOT") {
		return dir[len("$SYSROOT"):]
	}
	return dir
}

var (
	reDirectLinkerInput = re(`^(\./|[a-zA-Z0-9_/]).*\.(a|o|obj|dll|dylib|lib|so|tbd)$`)
	reLinkerLibFile     = re(`(\.(a|o|obj|dll|dylib|lib|so|tbd)$|\.so\.)`)
)

// isLocalLinkerPath reports whether path is local to the build (a relative
// path, or an absolute path within the working directory, module cache, or
// temp/work directory, such as from ${SRCDIR} expansion). Non-local host
// paths are outside the cmd/go threat model.
func isLocalLinkerPath(path string) bool {
	if !filepath.IsAbs(path) {
		return true
	}
	clean := filepath.Clean(path)
	for _, root := range []string{base.Cwd(), cfg.GOMODCACHE, os.TempDir()} {
		if root == "" {
			continue
		}
		if str.HasFilePathPrefix(clean, filepath.Clean(root)) {
			return true
		}
		if realRoot, err := filepath.EvalSymlinks(root); err == nil {
			if realClean, err := filepath.EvalSymlinks(clean); err == nil && str.HasFilePathPrefix(realClean, realRoot) {
				return true
			}
		}
	}
	return false
}

func checkLinkerSpecDir(source, dir string) error {
	return scanLinkerSearchDir(source, dir, false)
}

func checkLinkerLibDir(source, dir string) error {
	return scanLinkerSearchDir(source, dir, true)
}

// scanLinkerSearchDir walks the directory rooted at dir and looks for .spec files,
// which can inject arbitrary linker flags, or cause execution of arbitrary
// commands. When checkLibs is true, it also validates library and object files in
// the directory via checkLinkerInputFile to reject GNU ld linker scripts and
// objects with embedded linker options or LTO sections.
//
// When GCC evaluates %:include(libgomp.spec) or %:include(libsanitizer.spec),
// which can be caused by -fopenmp, -fopenacc, or -fsanitize=..., it looks for the specific
// spec file in a number of places, including the current working directory and
// sysroot or -L directories. GCC uses a search strategy that looks for target-triple
// directories and GCC version directories inside of these directories, but since
// we don't want to hard code those values, we use a much less targeted approach,
// and just walk the directory looking for a spec file. In theory this can cause
// failure when we find a spec file that GCC wouldn't load, but it seems better to
// be safe here.
func scanLinkerSearchDir(source, rawDir string, checkLibs bool) error {
	dir := trimLinkerSysrootPrefix(rawDir)
	if dir == "" || !isLocalLinkerPath(dir) {
		return nil
	}
	visited := make(map[string]bool)
	var checkTree func(root string) error
	checkTree = func(root string) error {
		if abs, err := filepath.Abs(root); err == nil {
			root = abs
		} else {
			root = filepath.Clean(root)
		}
		if !isLocalLinkerPath(root) {
			return nil
		}
		if resolved, err := filepath.EvalSymlinks(root); err == nil {
			root = resolved
		}
		if visited[root] || !isLocalLinkerPath(root) {
			return nil
		}
		visited[root] = true
		if fi, err := os.Stat(root); err != nil || !fi.IsDir() {
			return nil
		}
		return filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
			if err != nil {
				return nil
			}
			if d.IsDir() {
				if path != root && !isLocalLinkerPath(path) {
					return filepath.SkipDir
				}
				return nil
			}
			if d.Type()&os.ModeSymlink != 0 {
				if fi, err := os.Stat(path); err == nil && fi.IsDir() {
					return checkTree(path)
				}
			}
			lower := strings.ToLower(d.Name())
			if lower == "specs" || strings.HasSuffix(lower, ".spec") {
				return fmt.Errorf("disallowed GCC spec file in %s: %s", source, path)
			}
			if checkLibs && reLinkerLibFile.MatchString(lower) {
				if err := checkLinkerInputFile(source, path); err != nil {
					return err
				}
			}
			return nil
		})
	}
	if err := checkTree(dir); err != nil {
		return err
	}
	if !checkLibs {
		return nil
	}
	// GCC also searches multilib sibling directories (such as ../lib64)
	// relative to each -L directory. Because filepath.Join cleans ".."
	// lexically, if dir is a symlink its lexical parent and its resolved
	// parent on disk may differ; check both.
	absDir := filepath.Clean(dir)
	if abs, err := filepath.Abs(dir); err == nil {
		absDir = abs
	}
	bases := []string{absDir}
	if resolved, err := filepath.EvalSymlinks(absDir); err == nil {
		if abs, err := filepath.Abs(resolved); err == nil {
			resolved = abs
		}
		if resolved != absDir {
			bases = append(bases, resolved)
		}
	}
	for _, base := range bases {
		// Avoid walking host top-level /lib, /lib64, C:\lib, etc. when base is a
		// filesystem root or depth-1 directory such as /tmp, /app, or C:\app
		// (where filepath.Join(base, "..", "lib") would resolve to /lib or C:\lib).
		if parent := filepath.Dir(base); parent == base || filepath.Dir(parent) == parent {
			continue
		}
		for _, sub := range []string{"lib", "lib32", "lib64", "libx32"} {
			if err := checkTree(filepath.Join(base, "..", sub)); err != nil {
				return err
			}
		}
	}
	return nil
}

// checkLinkerInputFile verifies that an existing linker input or .syso file
// is a valid binary object/archive (or macOS .tbd stub) rather than a GNU ld
// linker script, and does not contain embedded linker flags (unallowlisted
// .drectve directives, LC_LINKER_OPTION, or LTO IR sections).
func checkLinkerInputFile(source, path string) error {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		return err
	}

	if strings.HasSuffix(strings.ToLower(path), ".tbd") && source != ".syso" {
		var hdr [256]byte
		n, _ := f.Read(hdr[:])
		s := strings.TrimSpace(string(hdr[:n]))
		if !strings.HasPrefix(s, "---") && !strings.HasPrefix(s, "{") {
			return fmt.Errorf("invalid .tbd file in %s: %s", source, path)
		}
		return nil
	}

	return checkLinkerObjectRegion(source, path, io.NewSectionReader(f, 0, fi.Size()), 0)
}

func isDisallowedLTOSection(name string) bool {
	return strings.HasPrefix(name, ".gnu.lto_") ||
		strings.HasPrefix(name, ".gnu.debuglto_") ||
		strings.HasPrefix(name, ".llvm.lto") ||
		name == ".llvmbc" ||
		name == ".llvmcmd"
}

func isCOFFMachine(mach uint16) bool {
	switch mach {
	case pe.IMAGE_FILE_MACHINE_I386,
		pe.IMAGE_FILE_MACHINE_AMD64,
		pe.IMAGE_FILE_MACHINE_ARM,
		pe.IMAGE_FILE_MACHINE_ARMNT,
		pe.IMAGE_FILE_MACHINE_ARM64,
		pe.IMAGE_FILE_MACHINE_LOONGARCH32,
		pe.IMAGE_FILE_MACHINE_LOONGARCH64,
		pe.IMAGE_FILE_MACHINE_RISCV32,
		pe.IMAGE_FILE_MACHINE_RISCV64,
		pe.IMAGE_FILE_MACHINE_RISCV128:
		return true
	}
	return false
}

// coffReaderAt wraps an io.SectionReader so debug/pe.NewFile's initial 96-byte
// DOS-header probe succeeds on valid headerless COFF .o files smaller than 96 bytes.
type coffReaderAt struct {
	r *io.SectionReader
}

func (c coffReaderAt) ReadAt(p []byte, off int64) (int, error) {
	if off == 0 && len(p) == 96 && c.r.Size() >= 20 && c.r.Size() < 96 {
		clear(p)
		if _, err := c.r.ReadAt(p[:c.r.Size()], 0); err != nil {
			return 0, err
		}
		return len(p), nil
	}
	return c.r.ReadAt(p, off)
}

func checkLinkerArchive(source, path string, r *io.SectionReader, depth int) error {
	if source == ".syso" {
		return fmt.Errorf("archive not allowed as .syso in %s: %s", source, path)
	}
	off := int64(8)
	memberIdx := 0
	slashCount := 0
	sawLongNames := false
	var arhdr [60]byte
	for off+60 <= r.Size() {
		if _, err := r.ReadAt(arhdr[:], off); err != nil {
			return fmt.Errorf("malformed archive in %s: %s", source, path)
		}
		if arhdr[58] != '`' || arhdr[59] != '\n' {
			return fmt.Errorf("malformed archive header in %s: %s", source, path)
		}
		name := strings.TrimSpace(string(arhdr[0:16]))
		sizeStr := strings.TrimSpace(string(arhdr[48:58]))
		sz, err := strconv.ParseInt(sizeStr, 10, 64)
		if err != nil || sz < 0 || off+60+sz > r.Size() {
			return fmt.Errorf("malformed archive member size in %s: %s", source, path)
		}
		off += 60
		switch {
		case name == "__.SYMDEF" || name == "__.SYMDEF SORTED":
			if memberIdx != 0 {
				return fmt.Errorf("unexpected archive symbol table member %q in %s: %s", name, source, path)
			}
		case name == "/":
			if memberIdx != slashCount || slashCount >= 2 || sawLongNames {
				return fmt.Errorf("unexpected archive symbol table member %q in %s: %s", name, source, path)
			}
			if slashCount == 0 {
				if err := checkArchiveFirstLinkerMember(source, path, r, off, sz); err != nil {
					return err
				}
			} else {
				if err := checkPESecondLinkerMember(source, path, r, off, sz); err != nil {
					return err
				}
			}
			slashCount++
		case name == "//":
			if memberIdx != slashCount || sawLongNames {
				return fmt.Errorf("unexpected archive string table member in %s: %s", source, path)
			}
			sawLongNames = true
			// Reject binary objects disguised as the "//" string table, which a
			// symbol offset in "/" could otherwise target and cause the linker to load.
			if sz >= 4 {
				if err := checkLinkerObjectRegion(source, path, io.NewSectionReader(r, off, sz), depth+1); err == nil {
					return fmt.Errorf("binary object disguised as archive string table in %s: %s", source, path)
				} else if strings.Contains(err.Error(), "disallowed") {
					return err
				}
			}
		case strings.HasPrefix(name, "#1/"):
			nameLen, err := strconv.ParseInt(name[3:], 10, 64)
			if err != nil || nameLen <= 0 || nameLen > 4096 || nameLen > sz {
				return fmt.Errorf("malformed BSD archive member in %s: %s", source, path)
			}
			nameBytes := make([]byte, nameLen)
			if _, err := io.ReadFull(io.NewSectionReader(r, off, nameLen), nameBytes); err != nil {
				return fmt.Errorf("malformed BSD archive member name in %s: %s", source, path)
			}
			realName := strings.TrimRight(string(nameBytes), "\x00")
			isSymdef := memberIdx == 0 && (realName == "__.SYMDEF" || realName == "__.SYMDEF SORTED")
			if !isSymdef {
				if sz <= nameLen {
					return fmt.Errorf("empty BSD archive member in %s: %s", source, path)
				}
				if err := checkLinkerObjectRegion(source, path, io.NewSectionReader(r, off+nameLen, sz-nameLen), depth+1); err != nil {
					return err
				}
			}
		default:
			if sz <= 0 {
				return fmt.Errorf("empty archive member in %s: %s", source, path)
			}
			if err := checkLinkerObjectRegion(source, path, io.NewSectionReader(r, off, sz), depth+1); err != nil {
				return err
			}
		}
		off += sz
		if off&1 != 0 {
			off++
		}
		memberIdx++
	}
	if off != r.Size() {
		return fmt.Errorf("truncated or trailing data in archive in %s: %s", source, path)
	}
	return nil
}

// checkArchiveFirstLinkerMember validates the SVR4/GNU/PE first "/" archive
// symbol table member. When resolving a symbol from "/", linkers seek directly
// to the recorded file offset without checking the target member's header name.
// The big-endian symbol count check (4+4*numSyms <= sz) rejects binary object
// headers starting with a non-zero byte (\x7fELF, Mach-O, MZ, COFF); we also
// explicitly reject the PE short import header magic (00 00 ff ff, which would
// otherwise parse as numSyms = 65535).
func checkArchiveFirstLinkerMember(source, path string, r *io.SectionReader, off, sz int64) error {
	if sz == 0 {
		return nil
	}
	if sz < 4 {
		return fmt.Errorf("malformed archive symbol table in %s: %s", source, path)
	}
	var shdr [4]byte
	if _, err := r.ReadAt(shdr[:], off); err != nil {
		return fmt.Errorf("malformed archive symbol table in %s: %s", source, path)
	}
	if shdr == [4]byte{0, 0, 0xff, 0xff} {
		return fmt.Errorf("binary object disguised as archive symbol table in %s: %s", source, path)
	}
	numSyms := int64(binary.BigEndian.Uint32(shdr[:]))
	if 4+4*numSyms > sz {
		return fmt.Errorf("malformed archive symbol table in %s: %s", source, path)
	}
	return nil
}

// checkPESecondLinkerMember validates the optional second "/" linker member in
// a Microsoft PE/COFF archive (which lists little-endian member count, member
// file offsets, symbol count, and symbol indices). The numMembers <= r.Size()/60
// bound rejects binary object headers disguised as a second "/" member.
func checkPESecondLinkerMember(source, path string, r *io.SectionReader, off, sz int64) error {
	if sz < 8 {
		return fmt.Errorf("malformed PE archive second linker member in %s: %s", source, path)
	}
	var mhdr [8]byte
	if _, err := r.ReadAt(mhdr[:], off); err != nil {
		return fmt.Errorf("malformed PE archive second linker member in %s: %s", source, path)
	}
	numMembers := int64(binary.LittleEndian.Uint32(mhdr[0:4]))
	if numMembers > r.Size()/60 || 4+4*numMembers+4 > sz {
		return fmt.Errorf("malformed PE archive second linker member in %s: %s", source, path)
	}
	var symCountBuf [4]byte
	if _, err := r.ReadAt(symCountBuf[:], off+4+4*numMembers); err != nil {
		return fmt.Errorf("malformed PE archive second linker member in %s: %s", source, path)
	}
	numSyms := int64(binary.LittleEndian.Uint32(symCountBuf[:]))
	if 4+4*numMembers+4+2*numSyms > sz {
		return fmt.Errorf("malformed PE archive second linker member in %s: %s", source, path)
	}
	return nil
}

// checkPEImportObject validates a PE short import library member
// (IMPORT_OBJECT_HEADER: Sig1=0x0000, Sig2=0xFFFF, Version=0 followed by
// null-terminated symbol and DLL name strings, which cannot contain .drectve).
func checkPEImportObject(source, path string, r *io.SectionReader) error {
	if r.Size() < 20 {
		return fmt.Errorf("invalid PE import object header in %s: %s", source, path)
	}
	var hdr [20]byte
	if _, err := r.ReadAt(hdr[:], 0); err != nil {
		return fmt.Errorf("invalid PE import object header in %s: %s", source, path)
	}
	mach := binary.LittleEndian.Uint16(hdr[6:8])
	if !isCOFFMachine(mach) {
		return fmt.Errorf("invalid PE import object machine %#x in %s: %s", mach, source, path)
	}
	sizeOfData := binary.LittleEndian.Uint32(hdr[12:16])
	if sizeOfData < 4 || sizeOfData > 1<<20 || 20+int64(sizeOfData) != r.Size() {
		return fmt.Errorf("invalid PE import object size in %s: %s", source, path)
	}
	flags := binary.LittleEndian.Uint16(hdr[18:20])
	typ := flags & 0x3
	nameType := (flags >> 2) & 0x7
	if typ > 2 || nameType > 3 || flags>>5 != 0 {
		return fmt.Errorf("invalid PE import object type flags %#x in %s: %s", flags, source, path)
	}
	data := make([]byte, sizeOfData)
	if _, err := io.ReadFull(io.NewSectionReader(r, 20, int64(sizeOfData)), data); err != nil {
		return fmt.Errorf("cannot read PE import object data in %s: %s: %v", source, path, err)
	}
	sep := bytes.IndexByte(data, 0)
	if sep <= 0 || sep >= len(data)-2 || data[len(data)-1] != 0 || bytes.IndexByte(data[sep+1:len(data)-1], 0) >= 0 {
		return fmt.Errorf("invalid PE import object strings in %s: %s", source, path)
	}
	return nil
}

func checkLinkerBigArchive(source, path string, r *io.SectionReader, depth int) error {
	if source == ".syso" {
		return fmt.Errorf("archive not allowed as .syso in %s: %s", source, path)
	}
	if r.Size() < 128 {
		return fmt.Errorf("malformed AIX big archive header in %s: %s", source, path)
	}
	var fhdr [128]byte
	if _, err := r.ReadAt(fhdr[:], 0); err != nil {
		return fmt.Errorf("malformed AIX big archive header in %s: %s", source, path)
	}
	parseDec := func(b []byte) (int64, error) {
		return strconv.ParseInt(strings.TrimSpace(string(b)), 10, 64)
	}
	off, err := parseDec(fhdr[68:88])
	if err != nil || off < 0 || (off > 0 && off < 128) || off > r.Size() {
		return fmt.Errorf("malformed AIX big archive first member offset in %s: %s", source, path)
	}
	if off == 0 {
		return nil
	}
	lastoff, err := parseDec(fhdr[88:108])
	if err != nil || lastoff < off || lastoff > r.Size() {
		return fmt.Errorf("malformed AIX big archive last member offset in %s: %s", source, path)
	}
	var mhdr [112]byte
	for {
		if off+112 > r.Size() {
			return fmt.Errorf("malformed AIX big archive member header in %s: %s", source, path)
		}
		if _, err := r.ReadAt(mhdr[:], off); err != nil {
			return fmt.Errorf("malformed AIX big archive member header in %s: %s", source, path)
		}
		sz, err := parseDec(mhdr[0:20])
		if err != nil || sz < 0 {
			return fmt.Errorf("malformed AIX big archive member size in %s: %s", source, path)
		}
		namlen, err := parseDec(mhdr[108:112])
		if err != nil || namlen <= 0 || namlen > 4096 {
			return fmt.Errorf("malformed AIX big archive member name length in %s: %s", source, path)
		}
		fileoff := off + 112 + namlen
		if fileoff&1 != 0 {
			fileoff++
		}
		var fmag [2]byte
		if fileoff+2 > r.Size() || sz > r.Size()-(fileoff+2) {
			return fmt.Errorf("malformed AIX big archive member bounds in %s: %s", source, path)
		}
		if _, err := r.ReadAt(fmag[:], fileoff); err != nil || fmag[0] != '`' || fmag[1] != '\n' {
			return fmt.Errorf("malformed AIX big archive member magic in %s: %s", source, path)
		}
		fileoff += 2
		if namlen > 0 {
			if sz == 0 {
				return fmt.Errorf("empty AIX big archive member in %s: %s", source, path)
			}
			if err := checkLinkerObjectRegion(source, path, io.NewSectionReader(r, fileoff, sz), depth+1); err != nil {
				return err
			}
		}
		if off == lastoff {
			break
		}
		nextoff, err := parseDec(mhdr[20:40])
		if err != nil || nextoff < fileoff+sz || nextoff > lastoff {
			return fmt.Errorf("malformed AIX big archive next member offset in %s: %s", source, path)
		}
		off = nextoff
	}
	return nil
}

// hasCOFFControlBeforeComment reports whether buf contains a non-whitespace C0
// control/null byte (or DEL) before any "/*" linker-script block comment can open.
// This rejects ASCII polyglot linker scripts starting with a COFF machine magic
// such as "db" (0x6264) or "dP" (0x5064) while allowing COFF objects with >= 256 sections.
func hasCOFFControlBeforeComment(buf []byte) bool {
	for i := 0; i < len(buf); i++ {
		if i+1 < len(buf) && buf[i] == '/' && buf[i+1] == '*' {
			return false
		}
		if buf[i] < 0x09 || (buf[i] > 0x0d && buf[i] < 0x20) || buf[i] == 0x7f {
			return true
		}
	}
	return false
}

func checkLinkerObjectRegion(source, path string, r *io.SectionReader, depth int) error {
	if depth > 4 {
		return fmt.Errorf("nested archive/fat object too deep in %s: %s", source, path)
	}
	var hdr [64]byte
	n, _ := io.ReadFull(r, hdr[:])
	if n < 4 {
		return fmt.Errorf("invalid object file in %s: %s", source, path)
	}
	buf := hdr[:n]

	if bytes.HasPrefix(buf, []byte("!<arch>\n")) {
		return checkLinkerArchive(source, path, r, depth)
	}
	if bytes.HasPrefix(buf, []byte("<bigaf>\n")) {
		return checkLinkerBigArchive(source, path, r, depth)
	}

	// Check PE short import library member (IMPORT_OBJECT_HEADER: Sig1=0x0000, Sig2=0xFFFF, Version=0).
	// Leading 0x00 bytes at offset 0..1 prevent GNU ld ASCII linker-script polyglots.
	if source != ".syso" && n >= 6 && binary.LittleEndian.Uint16(buf[0:2]) == 0x0000 && binary.LittleEndian.Uint16(buf[2:4]) == 0xffff && binary.LittleEndian.Uint16(buf[4:6]) == 0 {
		return checkPEImportObject(source, path, r)
	}

	// Check COFF object or PE image and validate .drectve and LTO sections via debug/pe.
	// Require buf[3] <= 1 for MZ (DOS header e_cblp < 0x200) and a non-whitespace control/null
	// byte in the 20-byte COFF FileHeader before any "/*" comment can open (plus structural
	// section/symbol table bounds checks), preventing an ASCII polyglot linker script starting
	// with "MZ" or an ASCII COFF machine magic (such as "db" or "dP") while allowing >= 256 sections.
	mach := binary.LittleEndian.Uint16(buf[0:2])
	isMZ := bytes.HasPrefix(buf, []byte("MZ")) && source != ".syso" && binary.LittleEndian.Uint16(buf[2:4]) < 0x200
	isCOFF := !isMZ && n >= 20 && isCOFFMachine(mach) && hasCOFFControlBeforeComment(buf[:20])
	if isCOFF || isMZ {
		var coffBase int64
		if isMZ {
			if n < 0x40 {
				return fmt.Errorf("invalid PE DOS header in %s: %s", source, path)
			}
			signoff := int64(binary.LittleEndian.Uint32(buf[0x3c:0x40]))
			if signoff < 0 || signoff+4 > r.Size() {
				return fmt.Errorf("invalid PE signature offset in %s: %s", source, path)
			}
			coffBase = signoff + 4
		}
		var rAt io.ReaderAt = r
		if !isMZ {
			rAt = coffReaderAt{r}
		}
		pf, err := pe.NewFile(rAt)
		if err != nil {
			return fmt.Errorf("invalid PE/COFF object in %s: %s: %v", source, path, err)
		}
		if !isCOFFMachine(pf.FileHeader.Machine) || (isMZ && pf.OptionalHeader == nil) || len(pf.Sections) == 0 {
			return fmt.Errorf("invalid PE/COFF header or missing sections in %s: %s", source, path)
		}
		hdrEnd := coffBase + 20 + int64(pf.FileHeader.SizeOfOptionalHeader) + int64(pf.FileHeader.NumberOfSections)*40
		if hdrEnd > r.Size() {
			return fmt.Errorf("truncated PE/COFF section table in %s: %s", source, path)
		}
		if pf.FileHeader.PointerToSymbolTable != 0 {
			symOff := int64(pf.FileHeader.PointerToSymbolTable)
			symEnd := symOff + int64(pf.FileHeader.NumberOfSymbols)*18
			if symOff < hdrEnd || symEnd > r.Size() {
				return fmt.Errorf("invalid PE/COFF symbol table bounds in %s: %s", source, path)
			}
		}
		for _, sec := range pf.Sections {
			if sec.Offset != 0 {
				secEnd := int64(sec.Offset) + int64(sec.Size)
				if secEnd > r.Size() || (sec.Size > 0 && int64(sec.Offset) < hdrEnd) {
					return fmt.Errorf("invalid PE/COFF section %q bounds in %s: %s", sec.Name, source, path)
				}
			}
			secName := strings.TrimSpace(sec.Name)
			if isDisallowedLTOSection(secName) {
				return fmt.Errorf("disallowed LTO section %q in %s: %s", secName, source, path)
			}
			if secName == ".drectve" {
				if sec.Size == 0 || sec.Size > 1<<20 {
					return fmt.Errorf("suspicious .drectve section size (%d) in %s: %s", sec.Size, source, path)
				}
				data, err := sec.Data()
				if err != nil {
					return fmt.Errorf("cannot read .drectve section in %s: %s: %v", source, path, err)
				}
				if err := checkDrectveDirectives(source, path, data); err != nil {
					return err
				}
			}
		}
		return nil
	}

	// Check Mach-O object via debug/macho and reject LC_LINKER_OPTION (0x2d) and LTO segments/sections.
	magicLE := binary.LittleEndian.Uint32(buf[0:4])
	magicBE := binary.BigEndian.Uint32(buf[0:4])
	if magicLE == macho.Magic32 || magicLE == macho.Magic64 || magicBE == macho.Magic32 || magicBE == macho.Magic64 {
		mf, err := macho.NewFile(r)
		if err != nil {
			return fmt.Errorf("invalid Mach-O object in %s: %s: %v", source, path, err)
		}
		for _, l := range mf.Loads {
			if raw := l.Raw(); len(raw) >= 4 && mf.ByteOrder.Uint32(raw[:4]) == 0x2d { // LC_LINKER_OPTION
				if err := checkMachOLinkerOption(source, path, mf.ByteOrder, raw); err != nil {
					return err
				}
			}
			if seg, ok := l.(*macho.Segment); ok {
				if seg.Name == "__LLVM" || seg.Name == "__GNU_LTO" {
					return fmt.Errorf("disallowed LTO segment %q in %s: %s", seg.Name, source, path)
				}
			}
		}
		for _, sec := range mf.Sections {
			if sec.Seg == "__LLVM" || sec.Seg == "__GNU_LTO" || isDisallowedLTOSection(sec.Name) {
				return fmt.Errorf("disallowed LTO section %s,%s in %s: %s", sec.Seg, sec.Name, source, path)
			}
		}
		return nil
	}

	// Check Universal (Fat) Mach-O (0xcafebabe) and validate each slice.
	// Note: slices inside a fat .a static library are !<arch>\n archives rather than
	// thin Mach-O files, so we read macho.FatArchHeader and recurse rather than calling macho.NewFatFile.
	if magicBE == macho.MagicFat {
		if n < 8 {
			return fmt.Errorf("truncated universal Mach-O header in %s: %s", source, path)
		}
		nfat := int(binary.BigEndian.Uint32(buf[4:8]))
		if nfat <= 0 || nfat > 32 {
			return fmt.Errorf("invalid universal Mach-O arch count in %s: %s", source, path)
		}
		sr := io.NewSectionReader(r, 8, int64(nfat)*20)
		for i := 0; i < nfat; i++ {
			var fa macho.FatArchHeader
			if err := binary.Read(sr, binary.BigEndian, &fa); err != nil {
				return fmt.Errorf("truncated universal Mach-O arch header in %s: %s", source, path)
			}
			sliceOff := int64(fa.Offset)
			sliceSz := int64(fa.Size)
			if sliceOff < 0 || sliceSz <= 0 || sliceOff+sliceSz > r.Size() {
				return fmt.Errorf("invalid universal Mach-O slice bounds in %s: %s", source, path)
			}
			if err := checkLinkerObjectRegion(source, path, io.NewSectionReader(r, sliceOff, sliceSz), depth+1); err != nil {
				return err
			}
		}
		return nil
	}

	// Check ELF object via debug/elf and reject GCC/LLVM LTO sections.
	if bytes.HasPrefix(buf, []byte(elf.ELFMAG)) {
		ef, err := elf.NewFile(r)
		if err != nil {
			return fmt.Errorf("invalid ELF object in %s: %s: %v", source, path, err)
		}
		if ef.Type == elf.ET_REL && len(ef.Sections) == 0 {
			return fmt.Errorf("ELF relocatable object missing section headers in %s: %s", source, path)
		}
		for _, sec := range ef.Sections {
			if sec.Type == 0x6fff4c01 || sec.Type == 0x6fff4c04 { // SHT_LLVM_LINKER_OPTIONS, SHT_LLVM_DEPENDENT_LIBRARIES
				return fmt.Errorf("disallowed ELF embedded linker options section in %s: %s", source, path)
			}
			if sec.Type == 0x6fff4c0c { // SHT_LLVM_LTO
				return fmt.Errorf("disallowed LLVM LTO section in %s: %s", source, path)
			}
			if isDisallowedLTOSection(sec.Name) {
				return fmt.Errorf("disallowed LTO section %q in %s: %s", sec.Name, source, path)
			}
		}
		return nil
	}

	// Check XCOFF object via internal/xcoff.
	magicBE16 := binary.BigEndian.Uint16(buf[0:2])
	if magicBE16 == xcoff.U802TOCMAGIC || magicBE16 == xcoff.U64_TOCMAGIC {
		if _, err := xcoff.NewFile(r); err != nil {
			return fmt.Errorf("invalid XCOFF object in %s: %s: %v", source, path, err)
		}
		return nil
	}

	// Check Plan 9 object via debug/plan9obj.
	if magicBE == plan9obj.Magic386 || magicBE == plan9obj.MagicAMD64 || magicBE == plan9obj.MagicARM {
		if _, err := plan9obj.NewFile(r); err != nil {
			return fmt.Errorf("invalid Plan 9 object in %s: %s: %v", source, path, err)
		}
		return nil
	}

	// Check WASM binary ("\x00asm\x01\x00\x00\x00").
	if bytes.HasPrefix(buf, []byte("\x00asm\x01\x00\x00\x00")) {
		return nil
	}
	return fmt.Errorf("invalid linker input file in %s: %s (unrecognized binary format)", source, path)
}

// checkMachOLinkerOption validates an LC_LINKER_OPTION (0x2d) load command,
// allowing safe library (-l<name>) and framework (-framework/-weak_framework <name>)
// auto-link directives emitted by Clang (-fmodules) and Rust/Swift staticlibs
// while rejecting arbitrary linker flags (such as -lto_library or -lazy_library).
func checkMachOLinkerOption(source, path string, bo binary.ByteOrder, raw []byte) error {
	if len(raw) < 12 {
		return fmt.Errorf("malformed LC_LINKER_OPTION in %s: %s", source, path)
	}
	count := bo.Uint32(raw[8:12])
	if count == 0 || count > 2 {
		return fmt.Errorf("disallowed LC_LINKER_OPTION in %s: %s", source, path)
	}
	rest := raw[12:]
	opts := make([]string, 0, count)
	for range count {
		nul := bytes.IndexByte(rest, 0)
		if nul <= 0 {
			return fmt.Errorf("disallowed LC_LINKER_OPTION in %s: %s", source, path)
		}
		opts = append(opts, string(rest[:nul]))
		rest = rest[nul+1:]
	}
	for _, b := range rest {
		if b != 0 {
			return fmt.Errorf("disallowed LC_LINKER_OPTION in %s: %s", source, path)
		}
	}
	switch len(opts) {
	case 1:
		arg := opts[0]
		if !strings.HasPrefix(arg, "-l") || len(arg) <= 2 || !load.SafeArg(arg[2:]) || strings.ContainsAny(arg[2:], `:=/\`) {
			return fmt.Errorf("disallowed LC_LINKER_OPTION %q in %s: %s", arg, source, path)
		}
		for _, re := range invalidLinkerFlags {
			if re.FindString(arg) == arg {
				return fmt.Errorf("disallowed LC_LINKER_OPTION %q in %s: %s", arg, source, path)
			}
		}
		return nil
	case 2:
		if (opts[0] == "-framework" || opts[0] == "-weak_framework") && load.SafeArg(opts[1]) {
			if err := checkLinkerNonPathNonFile(source, opts[0]+" "+opts[1], opts[1]); err == nil {
				return nil
			}
		}
		return fmt.Errorf("disallowed LC_LINKER_OPTION %q in %s: %s", strings.Join(opts, " "), source, path)
	}
	return fmt.Errorf("disallowed LC_LINKER_OPTION in %s: %s", source, path)
}

// checkDrectveDirectives validates the content of a COFF .drectve section,
// allowlisting known-safe MSVC and MinGW linker directives while rejecting
// anything that could inject dangerous linker behavior (e.g. --plugin, -T,
// --script, -L, --library-path, --dynamic-linker).
func checkDrectveDirectives(source, path string, data []byte) error {
	content := strings.TrimPrefix(string(data), "\xef\xbb\xbf")
	content = strings.TrimRight(content, "\x00 \t\r\n")
	if content == "" {
		return nil
	}
	tokens := tokenizeDrectve(content)
	for _, tok := range tokens {
		if !isAllowedDrectve(tok) {
			return fmt.Errorf("disallowed .drectve directive %q in %s: %s", tok, source, path)
		}
	}
	return nil
}

const drectveSpace = " \t\r\n\v\f\x00"

func tokenizeDrectve(s string) []string {
	var tokens []string
	s = strings.TrimLeft(s, drectveSpace)
	for len(s) > 0 {
		var tok strings.Builder
		inQuote := false
		i := 0
		for i < len(s) {
			if s[i] == '"' {
				inQuote = !inQuote
				i++
				continue
			}
			if !inQuote && strings.IndexByte(drectveSpace, s[i]) >= 0 {
				break
			}
			tok.WriteByte(s[i])
			i++
		}
		if tok.Len() > 0 {
			tokens = append(tokens, tok.String())
		}
		s = strings.TrimLeft(s[i:], drectveSpace)
	}
	return tokens
}

var reDrectveSafeName = lazyregexp.New(`^[a-zA-Z0-9_.][a-zA-Z0-9_.+\-@$=]*$`)

func isAllowedDrectve(tok string) bool {
	if strings.ContainsAny(tok, "\r\n\v\f\x00") {
		return false
	}
	norm := strings.ToUpper(tok)
	if strings.HasPrefix(norm, "-") && !strings.HasPrefix(norm, "--") {
		norm = "/" + norm[1:]
	}
	// MSVC, Clang, and MinGW directives: /FLAG:value or -flag:value
	if strings.HasPrefix(norm, "/DEFAULTLIB:") ||
		strings.HasPrefix(norm, "/DISALLOWLIB:") ||
		strings.HasPrefix(norm, "/EXPORT:") ||
		strings.HasPrefix(norm, "/INCLUDE:") ||
		strings.HasPrefix(norm, "/ALTERNATENAME:") ||
		strings.HasPrefix(norm, "/MERGE:") ||
		strings.HasPrefix(norm, "/SECTION:") ||
		strings.HasPrefix(norm, "/ALIGNCOMM:") ||
		strings.HasPrefix(norm, "/EXCLUDE-SYMBOLS:") ||
		strings.HasPrefix(norm, "/FAILIFMISMATCH:") ||
		strings.HasPrefix(norm, "/MANIFESTDEPENDENCY:") {
		val := tok[strings.IndexByte(tok, ':')+1:]
		val = strings.Trim(val, `"`)
		if val == "" {
			return false
		}
		if strings.ContainsAny(val, `/\`) && !strings.HasPrefix(norm, "/MANIFESTDEPENDENCY:") {
			return false
		}
		if (strings.HasPrefix(norm, "/DEFAULTLIB:") || strings.HasPrefix(norm, "/DISALLOWLIB:")) && (strings.Contains(val, ":") || strings.HasPrefix(val, "@")) {
			return false
		}
		if strings.Contains(val, "..") {
			return false
		}
		return true
	}
	// Flags with no value (or optional library name on NODEFAULTLIB) that are safe to pass through.
	if norm == "/NODEFAULTLIB" {
		return true
	}
	if strings.HasPrefix(norm, "/NODEFAULTLIB:") {
		val := strings.Trim(tok[strings.IndexByte(tok, ':')+1:], `"`)
		return val != "" && !strings.ContainsAny(val, `/\:`) && !strings.HasPrefix(val, "@") && !strings.Contains(val, "..")
	}
	if norm == "/SAFESEH" || norm == "/SAFESEH:NO" {
		return true
	}
	if norm == "/GUARD:CF" || norm == "/GUARD:NO" || norm == "/GUARD:EHCONT" || norm == "/GUARD:LONGJMP" {
		return true
	}
	if norm == "/DYNAMICBASE" || norm == "/DYNAMICBASE:NO" ||
		norm == "/NXCOMPAT" || norm == "/NXCOMPAT:NO" ||
		norm == "/CETCOMPAT" || norm == "/CETCOMPAT:NO" ||
		norm == "/HIGHENTROPYVA" || norm == "/HIGHENTROPYVA:NO" ||
		norm == "/LARGEADDRESSAWARE" || norm == "/LARGEADDRESSAWARE:NO" ||
		norm == "/TSAWARE" || norm == "/TSAWARE:NO" ||
		norm == "/EXPORT-ALL-SYMBOLS" || norm == "/EXCLUDE-ALL-SYMBOLS" {
		return true
	}
	// MinGW-style directives
	if strings.HasPrefix(tok, "-l") && len(tok) > 2 {
		libName := tok[2:]
		return reDrectveSafeName.MatchString(libName) && !strings.Contains(libName, "..")
	}
	switch tok {
	case "--enable-auto-import", "--disable-auto-import",
		"--enable-runtime-pseudo-reloc", "--disable-runtime-pseudo-reloc",
		"--enable-stdcall-fixup", "--disable-stdcall-fixup",
		"--export-all-symbols", "--exclude-all-symbols":
		return true
	}
	return false
}

func checkPkgConfigFlags(name, source string, list []string) error {
	checkOverrides := false
	return checkFlags(name, source, list, nil, validPkgConfigFlags, nil, nil, checkOverrides)
}

// checkCompilerFlagsForInternalLink returns an error if 'list'
// contains a flag or flags that may not be fully supported by
// internal linking (meaning that we should punt the link to the
// external linker).
func checkCompilerFlagsForInternalLink(name, source string, list []string) error {
	checkOverrides := false
	if err := checkFlags(name, source, list, invalidCompilerFlags, validCompilerFlags, nil, validCompilerFlagsWithNextArg, checkOverrides); err != nil {
		return err
	}
	// Currently the only flag on the allow list that causes problems
	// for the linker is "-flto"; check for it manually here.
	// Also check for -static/--static, which some toolchains accept
	// as a compiler flag.
	for _, fl := range list {
		if strings.HasPrefix(fl, "-flto") {
			return fmt.Errorf("flag %q triggers external linking", fl)
		}
		if fl == "-static" || fl == "--static" {
			return fmt.Errorf("flag %q triggers external linking", fl)
		}
	}
	return nil
}

// checkLinkerFlagsForInternalLink returns an error if 'list'
// contains linker flags that are not compatible with internal linking.
func checkLinkerFlagsForInternalLink(name, source string, list []string) error {
	checkOverrides := false
	if err := checkFlags(name, source, list, invalidLinkerFlags, validLinkerFlags, recombinedLinkerFlagsWithNextArg, validLinkerFlagsWithNextArg, checkOverrides); err != nil {
		return err
	}
	// Flags that force static linking require the external linker
	// to resolve libc symbols. See #77768.
	for _, fl := range list {
		if fl == "-static" || fl == "--static" ||
			fl == "-Wl,-static" || fl == "-Wl,--static" ||
			fl == "-Wl,-Bstatic" {
			return fmt.Errorf("flag %q triggers external linking", fl)
		}
	}
	return nil
}

func checkFlags(name, source string, list []string, invalid, valid []*lazyregexp.Regexp, recombineNext, validNext []string, checkOverrides bool) error {
	// Let users override rules with $CGO_CFLAGS_ALLOW, $CGO_CFLAGS_DISALLOW, etc.
	var (
		allow    *regexp.Regexp
		disallow *regexp.Regexp
	)
	if checkOverrides {
		if env := cfg.Getenv("CGO_" + name + "_ALLOW"); env != "" {
			r, err := regexp.Compile(env)
			if err != nil {
				return fmt.Errorf("parsing $CGO_%s_ALLOW: %v", name, err)
			}
			allow = r
		}
		if env := cfg.Getenv("CGO_" + name + "_DISALLOW"); env != "" {
			r, err := regexp.Compile(env)
			if err != nil {
				return fmt.Errorf("parsing $CGO_%s_DISALLOW: %v", name, err)
			}
			disallow = r
		}
	}

Args:
	for i := 0; i < len(list); i++ {
		arg := list[i]
		if disallow != nil && disallow.FindString(arg) == arg {
			goto Bad
		}
		if allow != nil && allow.FindString(arg) == arg {
			continue Args
		}
		for _, re := range invalid {
			if re.FindString(arg) == arg { // must be complete match
				goto Bad
			}
		}
		for _, re := range valid {
			if match := re.FindString(arg); match == arg { // must be complete match
				continue Args
			} else if strings.HasPrefix(arg, "-Wl,--push-state,") {
				// Examples for --push-state are written
				//     -Wl,--push-state,--as-needed
				// Support other commands in the same -Wl arg.
				args := strings.Split(arg, ",")
				for _, a := range args[1:] {
					a = "-Wl," + a
					var found bool
					for _, re := range valid {
						if re.FindString(a) == a {
							found = true
							break
						}
					}
					if !found {
						goto Bad
					}
					for _, re := range invalid {
						if re.FindString(a) == a {
							goto Bad
						}
					}
				}
				continue Args
			}
		}
		if slices.Contains(recombineNext, arg) && i+1 < len(list) {
			combined := arg + list[i+1]
			if disallow != nil && disallow.FindString(combined) == combined {
				goto Bad
			}
			if allow == nil || allow.FindString(combined) != combined {
				for _, re := range invalid {
					if re.FindString(combined) == combined {
						goto Bad
					}
				}
			}
		}
		for _, x := range validNext {
			if arg == x {
				if arg == "-x" {
					if i+1 < len(list) && validCompilerXLang.MatchString(list[i+1]) {
						i++
						continue Args
					}
					if i+1 < len(list) {
						return fmt.Errorf("invalid flag in %s: %s %s (see https://go.dev/s/invalidflag)", source, arg, list[i+1])
					}
					return fmt.Errorf("invalid flag in %s: %s without argument (see https://go.dev/s/invalidflag)", source, arg)
				}
				if strings.HasPrefix(arg, "-Wl,") {
					// Permit -Wl,-framework -Wl,name.
					if i+1 < len(list) &&
						strings.HasPrefix(list[i+1], "-Wl,") &&
						load.SafeArg(list[i+1][4:]) &&
						!strings.Contains(list[i+1][4:], ",") {
						i++
						continue Args
					}
					// Permit -Wl,-R /foo and -Wl,-R foo.
					if arg == "-Wl,-R" && i+1 < len(list) && load.SafeArg(list[i+1]) {
						i++
						continue Args
					}
					if i+1 < len(list) {
						return fmt.Errorf("invalid flag in %s: %s %s (see https://go.dev/s/invalidflag)", source, arg, list[i+1])
					}
					return fmt.Errorf("invalid flag in %s: %s without argument (see https://go.dev/s/invalidflag)", source, arg)
				}
				if arg == "-l" {
					if i+1 < len(list) && load.SafeArg(list[i+1]) && !strings.ContainsAny(list[i+1], `:=/\`) {
						i++
						continue Args
					}
					if i+1 < len(list) {
						return fmt.Errorf("invalid flag in %s: %s %s (see https://go.dev/s/invalidflag)", source, arg, list[i+1])
					}
					return fmt.Errorf("invalid flag in %s: %s without argument (see https://go.dev/s/invalidflag)", source, arg)
				}
				if i+1 < len(list) && load.SafeArg(list[i+1]) {
					i++
					continue Args
				}

				// Permit -I= /path, -I $SYSROOT.
				if i+1 < len(list) && arg == "-I" {
					if (strings.HasPrefix(list[i+1], "=") || strings.HasPrefix(list[i+1], "$SYSROOT")) &&
						load.SafeArg(list[i+1][1:]) {
						i++
						continue Args
					}
				}

				if i+1 < len(list) {
					return fmt.Errorf("invalid flag in %s: %s %s (see https://go.dev/s/invalidflag)", source, arg, list[i+1])
				}
				return fmt.Errorf("invalid flag in %s: %s without argument (see https://go.dev/s/invalidflag)", source, arg)
			}
		}
	Bad:
		return fmt.Errorf("invalid flag in %s: %s (see https://go.dev/s/invalidflag)", source, arg)
	}
	return nil
}
