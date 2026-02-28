// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package objabi

import (
	"bytes"
	"flag"
	"fmt"
	"internal/buildcfg"
	"io"
	"io/ioutil"
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
// argument is a filename with CR-or-CRLF-separated arguments. Each
// argument in the named files can also contain response file
// arguments. See Issue 18468.
//
// The returned slice 'out' aliases 'in' iff the input did not contain
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
			slurp, err := ioutil.ReadFile(s[1:])
			if err != nil {
				log.Fatal(err)
			}
			args := strings.Split(strings.TrimSpace(strings.Replace(string(slurp), "\r", "", -1)), "\n")
			for i, arg := range args {
				args[i] = DecodeArg(arg)
			}
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

func AddVersionFlag() {
	flag.Var(versionFlag{}, "V", "print version and exit")
}

var buildID string // filled in by linker

type versionFlag struct{}

func (versionFlag) IsBoolFlag() bool { return true }
func (versionFlag) Get() interface{} { return nil }
func (versionFlag) String() string   { return "" }
func (versionFlag) Set(s string) error {
	name := os.Args[0]
	name = name[strings.LastIndex(name, `/`)+1:]
	name = name[strings.LastIndex(name, `\`)+1:]
	name = strings.TrimSuffix(name, ".exe")

	p := ""

	if s == "goexperiment" {
		// test/run.go uses this to discover the full set of
		// experiment tags. Report everything.
		p = " X:" + strings.Join(buildcfg.Experiment.All(), ",")
	} else {
		// If the enabled experiments differ from the baseline,
		// include that difference.
		if goexperiment := buildcfg.Experiment.String(); goexperiment != "" {
			p = " X:" + goexperiment
		}
	}

	// The go command invokes -V=full to get a unique identifier
	// for this tool. It is assumed that the release version is sufficient
	// for releases, but during development we include the full
	// build ID of the binary, so that if the compiler is changed and
	// rebuilt, we notice and rebuild all packages.
	if s == "full" {
		if strings.HasPrefix(buildcfg.Version, "devel") {
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

func (c *count) Get() interface{} {
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

// DecodeArg decodes an argument.
//
// This function is public for testing with the parallel encoder.
func DecodeArg(arg string) string {
	// If no encoding, fastpath out.
	if !strings.ContainsAny(arg, "\\\n") {
		return arg
	}

	// We can't use strings.Builder as this must work at bootstrap.
	var b bytes.Buffer
	var wasBS bool
	for _, r := range arg {
		if wasBS {
			switch r {
			case '\\':
				b.WriteByte('\\')
			case 'n':
				b.WriteByte('\n')
			default:
				// This shouldn't happen. The only backslashes that reach here
				// should encode '\n' and '\\' exclusively.
				panic("badly formatted input")
			}
		} else if r == '\\' {
			wasBS = true
			continue
		} else {
			b.WriteRune(r)
		}
		wasBS = false
	}
	return b.String()
}

type debugField struct {
	name string
	help string
	val  interface{} // *int or *string
}

type DebugFlag struct {
	tab map[string]debugField
	any *bool

	debugSSA DebugSSA
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
func NewDebugFlag(debug interface{}, debugSSA DebugSSA) *DebugFlag {
	flag := &DebugFlag{
		tab:      make(map[string]debugField),
		debugSSA: debugSSA,
	}

	v := reflect.ValueOf(debug).Elem()
	t := v.Type()
	for i := 0; i < t.NumField(); i++ {
		f := t.Field(i)
		ptr := v.Field(i).Addr().Interface()
		if f.Name == "Any" {
			switch ptr := ptr.(type) {
			default:
				panic("debug.Any must have type bool")
			case *bool:
				flag.any = ptr
			}
			continue
		}
		name := strings.ToLower(f.Name)
		help := f.Tag.Get("help")
		if help == "" {
			panic(fmt.Sprintf("debug.%s is missing help text", f.Name))
		}
		switch ptr.(type) {
		default:
			panic(fmt.Sprintf("debug.%s has invalid type %v (must be int or string)", f.Name, f.Type))
		case *int, *string:
			// ok
		}
		flag.tab[name] = debugField{name, help, ptr}
	}

	return flag
}

func (f *DebugFlag) Set(debugstr string) error {
	if debugstr == "" {
		return nil
	}
	if f.any != nil {
		*f.any = true
	}
	for _, name := range strings.Split(debugstr, ",") {
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
				fmt.Printf("\t%-*s\t%s\n", maxLen, name, strings.Replace(help, "\n", nl, -1))
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
			default:
				panic("bad debugtab type")
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
				log.Fatalf(err)
			}
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
