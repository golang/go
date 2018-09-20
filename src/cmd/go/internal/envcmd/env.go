// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package envcmd implements the ``go env'' command.
package envcmd

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/cache"
	"cmd/go/internal/cfg"
	"cmd/go/internal/load"
	"cmd/go/internal/modload"
	"cmd/go/internal/work"
)

var CmdEnv = &base.Command{
	UsageLine: "go env [-json] [var ...]",
	Short:     "print Go environment information",
	Long: `
Env prints Go environment information.

By default env prints information as a shell script
(on Windows, a batch file). If one or more variable
names is given as arguments, env prints the value of
each named variable on its own line.

The -json flag prints the environment in JSON format
instead of as a shell script.

For more about environment variables, see 'go help environment'.
	`,
}

func init() {
	CmdEnv.Run = runEnv // break init cycle
}

var envJson = CmdEnv.Flag.Bool("json", false, "")

func MkEnv() []cfg.EnvVar {
	var b work.Builder
	b.Init()

	env := []cfg.EnvVar{
		{Name: "GOARCH", Value: cfg.Goarch},
		{Name: "GOBIN", Value: cfg.GOBIN},
		{Name: "GOCACHE", Value: cache.DefaultDir()},
		{Name: "GOEXE", Value: cfg.ExeSuffix},
		{Name: "GOFLAGS", Value: os.Getenv("GOFLAGS")},
		{Name: "GOHOSTARCH", Value: runtime.GOARCH},
		{Name: "GOHOSTOS", Value: runtime.GOOS},
		{Name: "GOOS", Value: cfg.Goos},
		{Name: "GOPATH", Value: cfg.BuildContext.GOPATH},
		{Name: "GOPROXY", Value: os.Getenv("GOPROXY")},
		{Name: "GORACE", Value: os.Getenv("GORACE")},
		{Name: "GOROOT", Value: cfg.GOROOT},
		{Name: "GOTMPDIR", Value: os.Getenv("GOTMPDIR")},
		{Name: "GOTOOLDIR", Value: base.ToolDir},
	}

	if work.GccgoBin != "" {
		env = append(env, cfg.EnvVar{Name: "GCCGO", Value: work.GccgoBin})
	} else {
		env = append(env, cfg.EnvVar{Name: "GCCGO", Value: work.GccgoName})
	}

	switch cfg.Goarch {
	case "arm":
		env = append(env, cfg.EnvVar{Name: "GOARM", Value: cfg.GOARM})
	case "386":
		env = append(env, cfg.EnvVar{Name: "GO386", Value: cfg.GO386})
	case "mips", "mipsle":
		env = append(env, cfg.EnvVar{Name: "GOMIPS", Value: cfg.GOMIPS})
	case "mips64", "mips64le":
		env = append(env, cfg.EnvVar{Name: "GOMIPS64", Value: cfg.GOMIPS64})
	}

	cc := cfg.DefaultCC(cfg.Goos, cfg.Goarch)
	if env := strings.Fields(os.Getenv("CC")); len(env) > 0 {
		cc = env[0]
	}
	cxx := cfg.DefaultCXX(cfg.Goos, cfg.Goarch)
	if env := strings.Fields(os.Getenv("CXX")); len(env) > 0 {
		cxx = env[0]
	}
	env = append(env, cfg.EnvVar{Name: "CC", Value: cc})
	env = append(env, cfg.EnvVar{Name: "CXX", Value: cxx})

	if cfg.BuildContext.CgoEnabled {
		env = append(env, cfg.EnvVar{Name: "CGO_ENABLED", Value: "1"})
	} else {
		env = append(env, cfg.EnvVar{Name: "CGO_ENABLED", Value: "0"})
	}

	return env
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
	if modload.Init(); modload.ModRoot != "" {
		gomod = filepath.Join(modload.ModRoot, "go.mod")
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

func runEnv(cmd *base.Command, args []string) {
	env := cfg.CmdEnv
	env = append(env, ExtraEnvVars()...)

	// Do we need to call ExtraEnvVarsCostly, which is a bit expensive?
	// Only if we're listing all environment variables ("go env")
	// or the variables being requested are in the extra list.
	needCostly := true
	if len(args) > 0 {
		needCostly = false
		for _, arg := range args {
			switch arg {
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
					fmt.Printf("%s='%s'\n", e.Name, strings.Replace(e.Value, "'", "''", -1))
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
		base.Fatalf("%s", err)
	}
}
