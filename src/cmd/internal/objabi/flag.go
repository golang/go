// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package objabi

import (
	"flag"
	"fmt"
	"internal/bisect"
	"internal/buildcfg"
	"io"
	"log"
	"os"
	"reflect"
	"sort"
	"strconv"
	"strings"
)

func Flagcount(name, usage string, val *int) {
	flag.Var((*count)(val), name, usage)
}

func Flagfn1(name, usage string, f func(string)) {
	flag.Var(fn1(f), name, usage)
}

func Flagprint(w io.Writer) {
	flag.CommandLine.SetOutput(w)
	flag.PrintDefaults()
}

func Flagparse(usage func()) {
	flag.Usage = usage
	os.Args = expandArgs(os.Args)
	flag.Parse()
}

// expandArgs expands "response files" arguments in the provided slice.
//
// A "response file" argument starts with '@' and the rest of that
// argument is a filename with arguments. Arguments are separated by
// whitespace, and can use single quotes (literal) or double quotes
// (with escape sequences). Each argument in the named files can also
// contain response file arguments. See Issue 77177.
//
// The returned slice 'out' aliases 'in' if the input did not contain
// any response file arguments.
//
// TODO: handle relative paths of recursive expansions in different directories?
// Is there a spec for this? Are relative paths allowed?
func expandArgs(in []string) (out []string) {
	// out is nil until we see a "@" argument.
	for i, s := range in {
		if strings.HasPrefix(s, "@") {
			if out == nil {
				out = make([]string, 0, len(in)*2)
				out = append(out, in[:i]...)
			}
			slurp, err := os.ReadFile(s[1:])
			if err != nil {
				log.Fatal(err)
			}
			args := ParseArgs(slurp)
			out = append(out, expandArgs(args)...)
		} else if out != nil {
			out = append(out, s)
		}
	}
	if out == nil {
		return in
	}
	return
}

// ParseArgs parses response file content into arguments using GCC-compatible rules.
// Arguments are separated by whitespace. Single quotes preserve content literally.
// Double quotes allow escape sequences: \\, \", \$, \`, and backslash-newline
// for line continuation (both LF and CRLF). Outside quotes, backslash escapes the
// next character, backslash-newline is line continuation (both LF and CRLF).
// We aim to follow GCC's buildargv implementation.
// Source code: https://github.com/gcc-mirror/gcc/blob/releases/gcc-15.2.0/libiberty/argv.c#L167
// Known deviations from GCC:
// - CRLF is treated as line continuation to be Windows-friendly; GCC only recognizes LF.
// - Obsolete \f and \v are not treated as whitespaces
// This function is public to test with cmd/go/internal/work.encodeArg
func ParseArgs(s []byte) []string {
	var args []string
	var arg strings.Builder
	hasArg := false // tracks if we've started an argument (for empty quotes)
	inSingleQuote := false
	inDoubleQuote := false
	i := 0

	for i < len(s) {
		c := s[i]

		if inSingleQuote {
			if c == '\'' {
				inSingleQuote = false
			} else {
				arg.WriteByte(c) // No escape processing in single quotes
			}
			i++
			continue
		}

		if inDoubleQuote {
			if c == '\\' && i+1 < len(s) {
				next := s[i+1]
				switch next {
				case '\\':
					arg.WriteByte('\\')
					i += 2
				case '"':
					arg.WriteByte('"')
					i += 2
				case '$':
					arg.WriteByte('$')
					i += 2
				case '`':
					arg.WriteByte('`')
					i += 2
				case '\n':
					// Line continuation - skip backslash and newline
					i += 2
				case '\r':
					// Line continuation for CRLF - skip backslash, CR, and LF
					if i+2 < len(s) && s[i+2] == '\n' {
						i += 3
					} else {
						arg.WriteByte(c)
						i++
					}
				default:
					// Unknown escape - keep backslash and char
					arg.WriteByte(c)
					i++
				}
			} else if c == '"' {
				inDoubleQuote = false
				i++
			} else {
				arg.WriteByte(c)
				i++
			}
			continue
		}

		// Normal mode (outside quotes)
		switch c {
		case ' ', '\t', '\n', '\r':
			if arg.Len() > 0 || hasArg {
				args = append(args, arg.String())
				arg.Reset()
				hasArg = false
			}
		case '\'':
			inSingleQuote = true
			hasArg = true // Empty quotes still produce an arg
		case '"':
			inDoubleQuote = true
			hasArg = true // Empty quotes still produce an arg
		case '\\':
			// Backslash escapes the next character outside quotes.
			// Backslash-newline is line continuation (handles both LF and CRLF).
			if i+1 < len(s) {
				next := s[i+1]
				if next == '\n' {
					i += 2
					continue
				}
				if next == '\r' && i+2 < len(s) && s[i+2] == '\n' {
					i += 3
					continue
				}
				// Backslash escapes the next character
				arg.WriteByte(next)
				hasArg = true
				i += 2
				continue
			}
			// Trailing backslash at end of input — consumed and discarded
			i++
			continue
		default:
			arg.WriteByte(c)
		}
		i++
	}

	// Don't forget the last argument
	if arg.Len() > 0 || hasArg {
		args = append(args, arg.String())
	}

	return args
}

func AddVersionFlag() {
	flag.Var(versionFlag{}, "V", "print version and exit")
}

var buildID string // filled in by linker

type versionFlag struct{}

func (versionFlag) IsBoolFlag() bool { return true }
func (versionFlag) Get() any         { return nil }
func (versionFlag) String() string   { return "" }
func (versionFlag) Set(s string) error {
	name := os.Args[0]
	name = name[strings.LastIndex(name, `/`)+1:]
	name = name[strings.LastIndex(name, `\`)+1:]
	name = strings.TrimSuffix(name, ".exe")

	p := ""

	// If the enabled experiments differ from the baseline,
	// include that difference.
	if goexperiment := buildcfg.Experiment.String(); goexperiment != "" {
		p = " X:" + goexperiment
	}

	// The go command invokes -V=full to get a unique identifier
	// for this tool. It is assumed that the release version is sufficient
	// for releases, but during development we include the full
	// build ID of the binary, so that if the compiler is changed and
	// rebuilt, we notice and rebuild all packages.
	if s == "full" {
		if strings.Contains(buildcfg.Version, "devel") {
			p += " buildID=" + buildID
		}
	}

	fmt.Printf("%s version %s%s\n", name, buildcfg.Version, p)
	os.Exit(0)
	return nil
}

// count is a flag.Value that is like a flag.Bool and a flag.Int.
// If used as -name, it increments the count, but -name=x sets the count.
// Used for verbose flag -v.
type count int

func (c *count) String() string {
	return fmt.Sprint(int(*c))
}

func (c *count) Set(s string) error {
	switch s {
	case "true":
		*c++
	case "false":
		*c = 0
	default:
		n, err := strconv.Atoi(s)
		if err != nil {
			return fmt.Errorf("invalid count %q", s)
		}
		*c = count(n)
	}
	return nil
}

func (c *count) Get() any {
	return int(*c)
}

func (c *count) IsBoolFlag() bool {
	return true
}

func (c *count) IsCountFlag() bool {
	return true
}

type fn1 func(string)

func (f fn1) Set(s string) error {
	f(s)
	return nil
}

func (f fn1) String() string { return "" }

type debugField struct {
	name         string
	help         string
	concurrentOk bool // true if this field/flag is compatible with concurrent compilation
	val          any  // *int or *string
}

type DebugFlag struct {
	tab          map[string]debugField
	concurrentOk *bool    // this is non-nil only for compiler's DebugFlags, but only compiler has concurrent:ok fields
	debugSSA     DebugSSA // this is non-nil only for compiler's DebugFlags.
}

// A DebugSSA function is called to set a -d ssa/... option.
// If nil, those options are reported as invalid options.
// If DebugSSA returns a non-empty string, that text is reported as a compiler error.
// If phase is "help", it should print usage information and terminate the process.
type DebugSSA func(phase, flag string, val int, valString string) string

// NewDebugFlag constructs a DebugFlag for the fields of debug, which
// must be a pointer to a struct.
//
// Each field of *debug is a different value, named for the lower-case of the field name.
// Each field must be an int or string and must have a `help` struct tag.
// There may be an "Any bool" field, which will be set if any debug flags are set.
//
// The returned flag takes a comma-separated list of settings.
// Each setting is name=value; for ints, name is short for name=1.
//
// If debugSSA is non-nil, any debug flags of the form ssa/... will be
// passed to debugSSA for processing.
func NewDebugFlag(debug any, debugSSA DebugSSA) *DebugFlag {
	flag := &DebugFlag{
		tab:      make(map[string]debugField),
		debugSSA: debugSSA,
	}

	v := reflect.ValueOf(debug).Elem()
	t := v.Type()
	for i := 0; i < t.NumField(); i++ {
		f := t.Field(i)
		ptr := v.Field(i).Addr().Interface()
		if f.Name == "ConcurrentOk" {
			switch ptr := ptr.(type) {
			default:
				panic("debug.ConcurrentOk must have type bool")
			case *bool:
				flag.concurrentOk = ptr
			}
			continue
		}
		name := strings.ToLower(f.Name)
		help := f.Tag.Get("help")
		if help == "" {
			panic(fmt.Sprintf("debug.%s is missing help text", f.Name))
		}
		concurrent := f.Tag.Get("concurrent")

		switch ptr.(type) {
		default:
			panic(fmt.Sprintf("debug.%s has invalid type %v (must be int, string, or *bisect.Matcher)", f.Name, f.Type))
		case *int, *string, **bisect.Matcher:
			// ok
		}
		flag.tab[name] = debugField{name, help, concurrent == "ok", ptr}
	}

	return flag
}

func (f *DebugFlag) Set(debugstr string) error {
	if debugstr == "" {
		return nil
	}
	for name := range strings.SplitSeq(debugstr, ",") {
		if name == "" {
			continue
		}
		// display help about the debug option itself and quit
		if name == "help" {
			fmt.Print(debugHelpHeader)
			maxLen, names := 0, []string{}
			if f.debugSSA != nil {
				maxLen = len("ssa/help")
			}
			for name := range f.tab {
				if len(name) > maxLen {
					maxLen = len(name)
				}
				names = append(names, name)
			}
			sort.Strings(names)
			// Indent multi-line help messages.
			nl := fmt.Sprintf("\n\t%-*s\t", maxLen, "")
			for _, name := range names {
				help := f.tab[name].help
				fmt.Printf("\t%-*s\t%s\n", maxLen, name, strings.ReplaceAll(help, "\n", nl))
			}
			if f.debugSSA != nil {
				// ssa options have their own help
				fmt.Printf("\t%-*s\t%s\n", maxLen, "ssa/help", "print help about SSA debugging")
			}
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

		if t, ok := f.tab[name]; ok {
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
			case **bisect.Matcher:
				var err error
				*vp, err = bisect.New(valstring)
				if err != nil {
					log.Fatalf("debug flag %v: %v", name, err)
				}
			default:
				panic("bad debugtab type")
			}
			// assembler DebugFlags don't have a ConcurrentOk field to reset, so check against that.
			if !t.concurrentOk && f.concurrentOk != nil {
				*f.concurrentOk = false
			}
		} else if f.debugSSA != nil && strings.HasPrefix(name, "ssa/") {
			// expect form ssa/phase/flag
			// e.g. -d=ssa/generic_cse/time
			// _ in phase name also matches space
			phase := name[4:]
			flag := "debug" // default flag is debug
			if i := strings.Index(phase, "/"); i >= 0 {
				flag = phase[i+1:]
				phase = phase[:i]
			}
			err := f.debugSSA(phase, flag, val, valstring)
			if err != "" {
				log.Fatal(err)
			}
			// Setting this false for -d=ssa/... preserves old behavior
			// of turning off concurrency for any debug flags.
			// It's not known for sure if this is necessary, but it is safe.
			*f.concurrentOk = false

		} else {
			return fmt.Errorf("unknown debug key %s\n", name)
		}
	}

	return nil
}

const debugHelpHeader = `usage: -d arg[,arg]* and arg is <key>[=<value>]

<key> is one of:

`

func (f *DebugFlag) String() string {
	return ""
}
