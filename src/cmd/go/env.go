// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/load"
	"fmt"
	"os"
	"runtime"
	"strings"
)

var cmdEnv = &base.Command{
	Run:       runEnv,
	UsageLine: "env [var ...]",
	Short:     "print Go environment information",
	Long: `
Env prints Go environment information.

By default env prints information as a shell script
(on Windows, a batch file).  If one or more variable
names is given as arguments,  env prints the value of
each named variable on its own line.
	`,
}

func mkEnv() []cfg.EnvVar {
	var b builder
	b.init()

	env := []cfg.EnvVar{
		{"GOARCH", cfg.Goarch},
		{"GOBIN", cfg.GOBIN},
		{"GOEXE", cfg.ExeSuffix},
		{"GOHOSTARCH", runtime.GOARCH},
		{"GOHOSTOS", runtime.GOOS},
		{"GOOS", cfg.Goos},
		{"GOPATH", cfg.BuildContext.GOPATH},
		{"GORACE", os.Getenv("GORACE")},
		{"GOROOT", cfg.GOROOT},
		{"GOTOOLDIR", base.ToolDir},

		// disable escape codes in clang errors
		{"TERM", "dumb"},
	}

	if gccgoBin != "" {
		env = append(env, cfg.EnvVar{"GCCGO", gccgoBin})
	} else {
		env = append(env, cfg.EnvVar{"GCCGO", gccgoName})
	}

	switch cfg.Goarch {
	case "arm":
		env = append(env, cfg.EnvVar{"GOARM", os.Getenv("GOARM")})
	case "386":
		env = append(env, cfg.EnvVar{"GO386", os.Getenv("GO386")})
	}

	cmd := b.gccCmd(".")
	env = append(env, cfg.EnvVar{"CC", cmd[0]})
	env = append(env, cfg.EnvVar{"GOGCCFLAGS", strings.Join(cmd[3:], " ")})
	cmd = b.gxxCmd(".")
	env = append(env, cfg.EnvVar{"CXX", cmd[0]})

	if cfg.BuildContext.CgoEnabled {
		env = append(env, cfg.EnvVar{"CGO_ENABLED", "1"})
	} else {
		env = append(env, cfg.EnvVar{"CGO_ENABLED", "0"})
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

// extraEnvVars returns environment variables that should not leak into child processes.
func extraEnvVars() []cfg.EnvVar {
	var b builder
	b.init()
	cppflags, cflags, cxxflags, fflags, ldflags := b.cflags(&load.Package{})
	return []cfg.EnvVar{
		{"PKG_CONFIG", b.pkgconfigCmd()},
		{"CGO_CFLAGS", strings.Join(cflags, " ")},
		{"CGO_CPPFLAGS", strings.Join(cppflags, " ")},
		{"CGO_CXXFLAGS", strings.Join(cxxflags, " ")},
		{"CGO_FFLAGS", strings.Join(fflags, " ")},
		{"CGO_LDFLAGS", strings.Join(ldflags, " ")},
	}
}

func runEnv(cmd *base.Command, args []string) {
	env := cfg.NewEnv
	env = append(env, extraEnvVars()...)
	if len(args) > 0 {
		for _, name := range args {
			fmt.Printf("%s\n", findEnv(env, name))
		}
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
