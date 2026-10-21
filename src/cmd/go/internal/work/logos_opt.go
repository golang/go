// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package work

import (
	"os"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/logos/config"
)

// logosOptFlag holds the raw -O level from the command line. Empty means
// unset: the level comes from logos.toml under logos and defaults to O1
// under plain go.
var logosOptFlag string

// logosInitGOAMD64 snapshots the user GOAMD64 at process startup,
// covering both the process environment and the GOENV file. main.invoke
// later injects every MkEnv value (including architecture defaults) into
// the process environment, so a post-invoke Getenv can never observe an
// unset GOAMD64; target_cpu precedence must read this snapshot instead.
var logosInitGOAMD64 = cfg.Getenv("GOAMD64")

// logosInitGOARM64 snapshots the user GOARM64 at process startup, for the
// same reason as logosInitGOAMD64: main.invoke injects architecture
// defaults before builds run.
var logosInitGOARM64 = cfg.Getenv("GOARM64")

// logosAddOptFlag registers the -O build flag. It carries no help text,
// so go help build stays byte-identical to upstream under both commands.
func logosAddOptFlag(cmd *base.Command) {
	cmd.Flag.StringVar(&logosOptFlag, "O", "", "")
}

// logosApplyOpt resolves the -O level and the target CPU, then appends
// the forced compile and link flags. It runs in BuildInit after
// cfgChangedEnv is built, so target_cpu entries join the tool
// environment that -x prints.
func logosApplyOpt() {
	file := logosOptFile()
	level := logosOptLevel(file)
	gc, ld := logosFlagsFor(level, logosSynthLevel(file, level))
	forcedGcflags = append(forcedGcflags, gc...)
	forcedLdflags = append(forcedLdflags, ld...)
	logosApplyTargetCPU(file)
}

// logosOptFile loads logos.toml for the module above the working
// directory. Plain go reads no logos.toml and gets nil. A broken file
// yields the defaults here: the package loader reports its file:line
// error for every package of the module, and file-argument builds ignore
// the file altogether.
func logosOptFile() *config.File {
	if !cfg.Logos {
		return nil
	}
	file, err := config.Load(cfg.LogosCWDModuleRoot())
	if err != nil {
		file, _ = config.Load("")
	}
	return file
}

// logosProgName reports the command name for user-facing -O errors.
func logosProgName() string {
	if cfg.Logos {
		return "logos"
	}
	return "go"
}

// logosValidOptLevel reports whether level names an accepted -O tier.
func logosValidOptLevel(level string) bool {
	switch level {
	case "0", "1", "2", "3", "s", "z":
		return true
	}
	return false
}

// logosOptLevel resolves the effective level: the -O flag wins, then the
// logos.toml opt_level under logos, then O1. File spellings carry the O
// ("O2"); the bare digit ("2") is used downstream.
func logosOptLevel(file *config.File) string {
	if logosOptFlag != "" {
		if !logosValidOptLevel(logosOptFlag) {
			base.Fatalf("%s build: invalid -O level %q (want 0, 1, 2, 3, s, z)", logosProgName(), logosOptFlag)
		}
		return logosOptFlag
	}
	if file != nil {
		return strings.TrimPrefix(file.Build.OptLevel, "O")
	}
	return "1"
}

// logosSynthLevel resolves the synthetic PGO level to pass as
// -d=logossynthpgo. An explicit key wins over the per-level default (O2
// standard, otherwise aggressive), and off maps to no flag. Only the O2
// and O3 flag sets carry the result: the other levels emit nothing.
func logosSynthLevel(file *config.File, level string) string {
	if file != nil && file.Build.SynthPGOSet {
		if file.Build.SyntheticPGO == "off" {
			return ""
		}
		return file.Build.SyntheticPGO
	}
	if level == "2" {
		return "standard"
	}
	return "aggressive"
}

// logosFlagsFor maps a level to forced compile and link flags. Os and Oz
// also pass the linker -logos.icf flag.
func logosFlagsFor(level, synth string) (gc, ld []string) {
	suffix := ""
	if synth != "" {
		suffix = ",logossynthpgo=" + synth
	}
	switch level {
	case "0":
		gc = []string{"-N", "-l"}
	case "2":
		gc = []string{"-d=inlbudget=120,variablemakethreshold=256" + suffix}
	case "3":
		gc = []string{"-d=inlbudget=400,inlrecursivebudget=80,variablemakethreshold=4096,logosspecialize=1" + suffix}
	case "s", "z":
		gc = []string{"-d=inlbudget=15"}
		ld = []string{"-logos.icf"}
	}
	return gc, ld
}

// logosApplyTargetCPU sets GOAMD64 and GOARM64 from target_cpu. An
// environment or GOENV setting wins over the file. Plain go keeps the
// upstream defaults. The levels come from config.ArchLevels, the single
// owner of the target_cpu selection.
func logosApplyTargetCPU(file *config.File) {
	if file == nil {
		return
	}
	amd64, arm64 := config.ArchLevels(file.Build.TargetCPU)
	logosSetArchLevel("GOAMD64", amd64)
	logosSetArchLevel("GOARM64", arm64)
}

// logosSetArchLevel sets one arch level when the startup environment
// leaves it unset, across the go command config, the process environment
// that tool subprocesses inherit, and the changed config environment that
// -x prints.
func logosSetArchLevel(key, val string) {
	if key == "GOAMD64" {
		if logosInitGOAMD64 != "" {
			return
		}
		cfg.GOAMD64 = val
	} else {
		if logosInitGOARM64 != "" {
			return
		}
		cfg.GOARM64 = val
	}
	os.Setenv(key, val)
	cfgChangedEnv = append(cfgChangedEnv, key+"="+val)
}
