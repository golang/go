// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package envcmd implements the ``go env'' command.
package envcmd

import (
	"encoding/json"
	"fmt"
	"go/build"
	"io/ioutil"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"unicode/utf8"

	"cmd/go/internal/base"
	"cmd/go/internal/cache"
	"cmd/go/internal/cfg"
	"cmd/go/internal/load"
	"cmd/go/internal/modload"
	"cmd/go/internal/work"
)

var CmdEnv = &base.Command{
	UsageLine: "go env [-json] [-u] [-w] [var ...]",
	Short:     "print Go environment information",
	Long: `
Env prints Go environment information.

By default env prints information as a shell script
(on Windows, a batch file). If one or more variable
names is given as arguments, env prints the value of
each named variable on its own line.

The -json flag prints the environment in JSON format
instead of as a shell script.

The -u flag requires one or more arguments and unsets
the default setting for the named environment variables,
if one has been set with 'go env -w'.

The -w flag requires one or more arguments of the
form NAME=VALUE and changes the default settings
of the named environment variables to the given values.

For more about environment variables, see 'go help environment'.
	`,
}

func init() {
	CmdEnv.Run = runEnv // break init cycle
}

var (
	envJson = CmdEnv.Flag.Bool("json", false, "")
	envU    = CmdEnv.Flag.Bool("u", false, "")
	envW    = CmdEnv.Flag.Bool("w", false, "")
)

func MkEnv() []cfg.EnvVar {
	var b work.Builder
	b.Init()

	envFile, _ := cfg.EnvFile()
	env := []cfg.EnvVar{
		{Name: "GO111MODULE", Value: cfg.Getenv("GO111MODULE")},
		{Name: "GOARCH", Value: cfg.Goarch},
		{Name: "GOBIN", Value: cfg.GOBIN},
		{Name: "GOCACHE", Value: cache.DefaultDir()},
		{Name: "GOENV", Value: envFile},
		{Name: "GOEXE", Value: cfg.ExeSuffix},
		{Name: "GOFLAGS", Value: cfg.Getenv("GOFLAGS")},
		{Name: "GOHOSTARCH", Value: runtime.GOARCH},
		{Name: "GOHOSTOS", Value: runtime.GOOS},
		{Name: "GOINSECURE", Value: cfg.GOINSECURE},
		{Name: "GONOPROXY", Value: cfg.GONOPROXY},
		{Name: "GONOSUMDB", Value: cfg.GONOSUMDB},
		{Name: "GOOS", Value: cfg.Goos},
		{Name: "GOPATH", Value: cfg.BuildContext.GOPATH},
		{Name: "GOPRIVATE", Value: cfg.GOPRIVATE},
		{Name: "GOPROXY", Value: cfg.GOPROXY},
		{Name: "GOROOT", Value: cfg.GOROOT},
		{Name: "GOSUMDB", Value: cfg.GOSUMDB},
		{Name: "GOTMPDIR", Value: cfg.Getenv("GOTMPDIR")},
		{Name: "GOTOOLDIR", Value: base.ToolDir},
	}

	if work.GccgoBin != "" {
		env = append(env, cfg.EnvVar{Name: "GCCGO", Value: work.GccgoBin})
	} else {
		env = append(env, cfg.EnvVar{Name: "GCCGO", Value: work.GccgoName})
	}

	key, val := cfg.GetArchEnv()
	if key != "" {
		env = append(env, cfg.EnvVar{Name: key, Value: val})
	}

	cc := cfg.DefaultCC(cfg.Goos, cfg.Goarch)
	if env := strings.Fields(cfg.Getenv("CC")); len(env) > 0 {
		cc = env[0]
	}
	cxx := cfg.DefaultCXX(cfg.Goos, cfg.Goarch)
	if env := strings.Fields(cfg.Getenv("CXX")); len(env) > 0 {
		cxx = env[0]
	}
	env = append(env, cfg.EnvVar{Name: "AR", Value: envOr("AR", "ar")})
	env = append(env, cfg.EnvVar{Name: "CC", Value: cc})
	env = append(env, cfg.EnvVar{Name: "CXX", Value: cxx})

	if cfg.BuildContext.CgoEnabled {
		env = append(env, cfg.EnvVar{Name: "CGO_ENABLED", Value: "1"})
	} else {
		env = append(env, cfg.EnvVar{Name: "CGO_ENABLED", Value: "0"})
	}

	return env
}

func envOr(name, def string) string {
	val := cfg.Getenv(name)
	if val != "" {
		return val
	}
	return def
}

func findEnv(env []cfg.EnvVar, name string) string {
	for _, e := range env {
		if e.Name == name {
			return e.Value
		}
	}
	return ""
}

// ExtraEnvVars returns environment variables that should not leak into child processes.
func ExtraEnvVars() []cfg.EnvVar {
	gomod := ""
	if modload.HasModRoot() {
		gomod = filepath.Join(modload.ModRoot(), "go.mod")
	} else if modload.Enabled() {
		gomod = os.DevNull
	}
	return []cfg.EnvVar{
		{Name: "GOMOD", Value: gomod},
	}
}

// ExtraEnvVarsCostly returns environment variables that should not leak into child processes
// but are costly to evaluate.
func ExtraEnvVarsCostly() []cfg.EnvVar {
	var b work.Builder
	b.Init()
	cppflags, cflags, cxxflags, fflags, ldflags, err := b.CFlags(&load.Package{})
	if err != nil {
		// Should not happen - b.CFlags was given an empty package.
		fmt.Fprintf(os.Stderr, "go: invalid cflags: %v\n", err)
		return nil
	}
	cmd := b.GccCmd(".", "")

	return []cfg.EnvVar{
		// Note: Update the switch in runEnv below when adding to this list.
		{Name: "CGO_CFLAGS", Value: strings.Join(cflags, " ")},
		{Name: "CGO_CPPFLAGS", Value: strings.Join(cppflags, " ")},
		{Name: "CGO_CXXFLAGS", Value: strings.Join(cxxflags, " ")},
		{Name: "CGO_FFLAGS", Value: strings.Join(fflags, " ")},
		{Name: "CGO_LDFLAGS", Value: strings.Join(ldflags, " ")},
		{Name: "PKG_CONFIG", Value: b.PkgconfigCmd()},
		{Name: "GOGCCFLAGS", Value: strings.Join(cmd[3:], " ")},
	}
}

// argKey returns the KEY part of the arg KEY=VAL, or else arg itself.
func argKey(arg string) string {
	i := strings.Index(arg, "=")
	if i < 0 {
		return arg
	}
	return arg[:i]
}

func runEnv(cmd *base.Command, args []string) {
	if *envJson && *envU {
		base.Fatalf("go env: cannot use -json with -u")
	}
	if *envJson && *envW {
		base.Fatalf("go env: cannot use -json with -w")
	}
	if *envU && *envW {
		base.Fatalf("go env: cannot use -u with -w")
	}
	env := cfg.CmdEnv
	env = append(env, ExtraEnvVars()...)

	// Do we need to call ExtraEnvVarsCostly, which is a bit expensive?
	// Only if we're listing all environment variables ("go env")
	// or the variables being requested are in the extra list.
	needCostly := true
	if len(args) > 0 {
		needCostly = false
		for _, arg := range args {
			switch argKey(arg) {
			case "CGO_CFLAGS",
				"CGO_CPPFLAGS",
				"CGO_CXXFLAGS",
				"CGO_FFLAGS",
				"CGO_LDFLAGS",
				"PKG_CONFIG",
				"GOGCCFLAGS":
				needCostly = true
			}
		}
	}
	if needCostly {
		env = append(env, ExtraEnvVarsCostly()...)
	}

	if *envW {
		// Process and sanity-check command line.
		if len(args) == 0 {
			base.Fatalf("go env -w: no KEY=VALUE arguments given")
		}
		osEnv := make(map[string]string)
		for _, e := range cfg.OrigEnv {
			if i := strings.Index(e, "="); i >= 0 {
				osEnv[e[:i]] = e[i+1:]
			}
		}
		add := make(map[string]string)
		for _, arg := range args {
			i := strings.Index(arg, "=")
			if i < 0 {
				base.Fatalf("go env -w: arguments must be KEY=VALUE: invalid argument: %s", arg)
			}
			key, val := arg[:i], arg[i+1:]
			if err := checkEnvWrite(key, val); err != nil {
				base.Fatalf("go env -w: %v", err)
			}
			if _, ok := add[key]; ok {
				base.Fatalf("go env -w: multiple values for key: %s", key)
			}
			add[key] = val
			if osVal := osEnv[key]; osVal != "" && osVal != val {
				fmt.Fprintf(os.Stderr, "warning: go env -w %s=... does not override conflicting OS environment variable\n", key)
			}
		}

		goos, okGOOS := add["GOOS"]
		goarch, okGOARCH := add["GOARCH"]
		if okGOOS || okGOARCH {
			if !okGOOS {
				goos = cfg.Goos
			}
			if !okGOARCH {
				goarch = cfg.Goarch
			}
			if err := work.CheckGOOSARCHPair(goos, goarch); err != nil {
				base.Fatalf("go env -w: %v", err)
			}
		}

		updateEnvFile(add, nil)
		return
	}

	if *envU {
		// Process and sanity-check command line.
		if len(args) == 0 {
			base.Fatalf("go env -u: no arguments given")
		}
		del := make(map[string]bool)
		for _, arg := range args {
			if err := checkEnvWrite(arg, ""); err != nil {
				base.Fatalf("go env -u: %v", err)
			}
			del[arg] = true
		}
		if del["GOOS"] || del["GOARCH"] {
			goos, goarch := cfg.Goos, cfg.Goarch
			if del["GOOS"] {
				goos = getOrigEnv("GOOS")
				if goos == "" {
					goos = build.Default.GOOS
				}
			}
			if del["GOARCH"] {
				goarch = getOrigEnv("GOARCH")
				if goarch == "" {
					goarch = build.Default.GOARCH
				}
			}
			if err := work.CheckGOOSARCHPair(goos, goarch); err != nil {
				base.Fatalf("go env -u: %v", err)
			}
		}
		updateEnvFile(nil, del)
		return
	}

	if len(args) > 0 {
		if *envJson {
			var es []cfg.EnvVar
			for _, name := range args {
				e := cfg.EnvVar{Name: name, Value: findEnv(env, name)}
				es = append(es, e)
			}
			printEnvAsJSON(es)
		} else {
			for _, name := range args {
				fmt.Printf("%s\n", findEnv(env, name))
			}
		}
		return
	}

	if *envJson {
		printEnvAsJSON(env)
		return
	}

	for _, e := range env {
		if e.Name != "TERM" {
			switch runtime.GOOS {
			default:
				fmt.Printf("%s=\"%s\"\n", e.Name, e.Value)
			case "plan9":
				if strings.IndexByte(e.Value, '\x00') < 0 {
					fmt.Printf("%s='%s'\n", e.Name, strings.ReplaceAll(e.Value, "'", "''"))
				} else {
					v := strings.Split(e.Value, "\x00")
					fmt.Printf("%s=(", e.Name)
					for x, s := range v {
						if x > 0 {
							fmt.Printf(" ")
						}
						fmt.Printf("%s", s)
					}
					fmt.Printf(")\n")
				}
			case "windows":
				fmt.Printf("set %s=%s\n", e.Name, e.Value)
			}
		}
	}
}

func printEnvAsJSON(env []cfg.EnvVar) {
	m := make(map[string]string)
	for _, e := range env {
		if e.Name == "TERM" {
			continue
		}
		m[e.Name] = e.Value
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "\t")
	if err := enc.Encode(m); err != nil {
		base.Fatalf("go env -json: %s", err)
	}
}

func getOrigEnv(key string) string {
	for _, v := range cfg.OrigEnv {
		if strings.HasPrefix(v, key+"=") {
			return strings.TrimPrefix(v, key+"=")
		}
	}
	return ""
}

func checkEnvWrite(key, val string) error {
	switch key {
	case "GOEXE", "GOGCCFLAGS", "GOHOSTARCH", "GOHOSTOS", "GOMOD", "GOTOOLDIR":
		return fmt.Errorf("%s cannot be modified", key)
	case "GOENV":
		return fmt.Errorf("%s can only be set using the OS environment", key)
	}

	// To catch typos and the like, check that we know the variable.
	if !cfg.CanGetenv(key) {
		return fmt.Errorf("unknown go command variable %s", key)
	}

	// Some variables can only have one of a few valid values. If set to an
	// invalid value, the next cmd/go invocation might fail immediately,
	// even 'go env -w' itself.
	switch key {
	case "GO111MODULE":
		switch val {
		case "", "auto", "on", "off":
		default:
			return fmt.Errorf("invalid %s value %q", key, val)
		}
	case "GOPATH":
		if strings.HasPrefix(val, "~") {
			return fmt.Errorf("GOPATH entry cannot start with shell metacharacter '~': %q", val)
		}
		if !filepath.IsAbs(val) && val != "" {
			return fmt.Errorf("GOPATH entry is relative; must be absolute path: %q", val)
		}
	}

	if !utf8.ValidString(val) {
		return fmt.Errorf("invalid UTF-8 in %s=... value", key)
	}
	if strings.Contains(val, "\x00") {
		return fmt.Errorf("invalid NUL in %s=... value", key)
	}
	if strings.ContainsAny(val, "\v\r\n") {
		return fmt.Errorf("invalid newline in %s=... value", key)
	}
	return nil
}

func updateEnvFile(add map[string]string, del map[string]bool) {
	file, err := cfg.EnvFile()
	if file == "" {
		base.Fatalf("go env: cannot find go env config: %v", err)
	}
	data, err := ioutil.ReadFile(file)
	if err != nil && (!os.IsNotExist(err) || len(add) == 0) {
		base.Fatalf("go env: reading go env config: %v", err)
	}

	lines := strings.SplitAfter(string(data), "\n")
	if lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	} else {
		lines[len(lines)-1] += "\n"
	}

	// Delete all but last copy of any duplicated variables,
	// since the last copy is the one that takes effect.
	prev := make(map[string]int)
	for l, line := range lines {
		if key := lineToKey(line); key != "" {
			if p, ok := prev[key]; ok {
				lines[p] = ""
			}
			prev[key] = l
		}
	}

	// Add variables (go env -w). Update existing lines in file if present, add to end otherwise.
	for key, val := range add {
		if p, ok := prev[key]; ok {
			lines[p] = key + "=" + val + "\n"
			delete(add, key)
		}
	}
	for key, val := range add {
		lines = append(lines, key+"="+val+"\n")
	}

	// Delete requested variables (go env -u).
	for key := range del {
		if p, ok := prev[key]; ok {
			lines[p] = ""
		}
	}

	// Sort runs of KEY=VALUE lines
	// (that is, blocks of lines where blocks are separated
	// by comments, blank lines, or invalid lines).
	start := 0
	for i := 0; i <= len(lines); i++ {
		if i == len(lines) || lineToKey(lines[i]) == "" {
			sortKeyValues(lines[start:i])
			start = i + 1
		}
	}

	data = []byte(strings.Join(lines, ""))
	err = ioutil.WriteFile(file, data, 0666)
	if err != nil {
		// Try creating directory.
		os.MkdirAll(filepath.Dir(file), 0777)
		err = ioutil.WriteFile(file, data, 0666)
		if err != nil {
			base.Fatalf("go env: writing go env config: %v", err)
		}
	}
}

// lineToKey returns the KEY part of the line KEY=VALUE or else an empty string.
func lineToKey(line string) string {
	i := strings.Index(line, "=")
	if i < 0 || strings.Contains(line[:i], "#") {
		return ""
	}
	return line[:i]
}

// sortKeyValues sorts a sequence of lines by key.
// It differs from sort.Strings in that GO386= sorts after GO=.
func sortKeyValues(lines []string) {
	sort.Slice(lines, func(i, j int) bool {
		return lineToKey(lines[i]) < lineToKey(lines[j])
	})
}
