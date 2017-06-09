// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package envcmd implements the ``go env'' command.
package envcmd

import (
	"encoding/json"
	"fmt"
	"os"
	"runtime"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/load"
	"cmd/go/internal/work"
)

var CmdEnv = &base.Command{
	UsageLine: "env [-json] [var ...]",
	Short:     "print Go environment information",
	Long: `
Env prints Go environment information.

By default env prints information as a shell script
(on Windows, a batch file). If one or more variable
names is given as arguments, env prints the value of
each named variable on its own line.

The -json flag prints the environment in JSON format
instead of as a shell script.
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
		{Name: "GOEXE", Value: cfg.ExeSuffix},
		{Name: "GOHOSTARCH", Value: runtime.GOARCH},
		{Name: "GOHOSTOS", Value: runtime.GOOS},
		{Name: "GOOS", Value: cfg.Goos},
		{Name: "GOPATH", Value: cfg.BuildContext.GOPATH},
		{Name: "GORACE", Value: os.Getenv("GORACE")},
		{Name: "GOROOT", Value: cfg.GOROOT},
		{Name: "GOTOOLDIR", Value: base.ToolDir},

		// disable escape codes in clang errors
		{Name: "TERM", Value: "dumb"},
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
	}

	cmd := b.GccCmd(".")
	env = append(env, cfg.EnvVar{Name: "CC", Value: cmd[0]})
	env = append(env, cfg.EnvVar{Name: "GOGCCFLAGS", Value: strings.Join(cmd[3:], " ")})
	cmd = b.GxxCmd(".")
	env = append(env, cfg.EnvVar{Name: "CXX", Value: cmd[0]})

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
	var b work.Builder
	b.Init()
	cppflags, cflags, cxxflags, fflags, ldflags := b.CFlags(&load.Package{})
	return []cfg.EnvVar{
		{Name: "CGO_CFLAGS", Value: strings.Join(cflags, " ")},
		{Name: "CGO_CPPFLAGS", Value: strings.Join(cppflags, " ")},
		{Name: "CGO_CXXFLAGS", Value: strings.Join(cxxflags, " ")},
		{Name: "CGO_FFLAGS", Value: strings.Join(fflags, " ")},
		{Name: "CGO_LDFLAGS", Value: strings.Join(ldflags, " ")},
		{Name: "PKG_CONFIG", Value: b.PkgconfigCmd()},
	}
}

func runEnv(cmd *base.Command, args []string) {
	env := cfg.CmdEnv
	env = append(env, ExtraEnvVars()...)
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
