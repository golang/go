// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cfg

import (
	"fmt"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"strings"
	"sync"

	"cmd/go/internal/logos/config"
	"cmd/internal/pathcache"
)

// Logos reports whether the current process was invoked as the logos
// command. The logos command is the same binary as the go command,
// selected by argv[0]: the base name of os.Args[0] with a ".exe"
// suffix removed must equal "logos". It is a package-level initializer,
// so defaultContext observes it when BuildContext is first built.
var Logos = strings.TrimSuffix(filepath.Base(os.Args[0]), ".exe") == "logos"

// Pedantic opts in to pedantic diagnostics for logos builds. It is set
// by the logos-only "logos build --pedantic" flag (registered in
// cmd/go/internal/logos.ConfigureCommand) and overrides the logos.toml
// pedantic policy (M9); this slice only stores it. Plain go never sets
// it: the flag is undefined there, so "go help build" keeps the
// upstream text.
var Pedantic bool

// PedanticSet reports whether --pedantic appeared on the command line, so
// --pedantic=false can override a pedantic key of logos.toml.
var PedanticSet bool

// LogosJSON is set by the logos-only --json-diagnostics flag of logos build
// and logos lint: compile diagnostics print as one JSON object per line
// (SPEC Sec 16). Plain go never sets it.
var LogosJSON bool

// BuildWasmLoader is the value of the --wasm-loader build flag (SPEC Sec
// 14.2): when true and the build writes a .wasm output with -o, the build
// also writes the host loader beside it. The flag defaults to true under
// logos and to false under plain go until the run.bash flip (R-U1), and it
// carries no help text, so "go help build" keeps the upstream text.
var BuildWasmLoader bool

// WasmLoader is the text of the host loader a wasm build writes beside its
// output. cmd/go/internal/logos sets it at program start, because that
// package imports the packages that import work and so work cannot import
// it.
var WasmLoader string

// LogosLintCmd is set while logos lint runs: every main-module package is
// linted by its own lint action, so an ordinary compile in the same run
// passes no lint policy to the compiler.
var LogosLintCmd bool

// BuildGC is the value of the -gc flag (SPEC Sec 19), and BuildGCSet reports
// whether the flag appeared, so -gc= can be told from an absent flag.
var (
	BuildGC    string
	BuildGCSet bool
)

// LogosGC is the collector mode the build selected, checked and defaulted by
// work.BuildInit: the -gc flag, else logos.toml [build] gc under logos, else
// the logos default. It is empty when plain go selected no mode, which keeps
// the upstream collector and compiles with write barriers.
var LogosGC string

// logosInitCC snapshots the user CC at process startup, covering both the
// process environment and the GOENV file. main.invoke later injects every
// MkEnv value (including a CC default) into the process environment, so a
// post-invoke Getenv can never observe an unset CC; the zig fallback and
// any other consumer must read this snapshot instead.
var logosInitCC = Getenv("CC")

// logosInitCXX snapshots the user CXX at process startup, covering both the
// process environment and the GOENV file, for the same reason as
// logosInitCC: the zig c++ fallback must tell an explicit user setting
// (including `go env -w CXX=...`) from an injected MkEnv default.
var logosInitCXX = Getenv("CXX")

// logosInitCGOEnabled snapshots the user CGO_ENABLED at process startup,
// covering both the process environment and the GOENV file. main.invoke
// injects the effective CGO_ENABLED into the process environment before
// commands run, so per-package precedence must read this snapshot instead
// of the live environment; see StartupCgoEnabled.
var logosInitCGOEnabled = Getenv("CGO_ENABLED")

// logosModuleKindCache memoizes module-kind results by module root for the
// life of the process, so repeated policy refreshes never re-walk a root.
var logosModuleKindCache = struct {
	sync.Mutex
	byRoot map[string]bool
}{byRoot: make(map[string]bool)}

// logosToolTags returns tags plus the "logos" tool tag in logos mode.
// defaultContext calls it, so every rebuilt BuildContext keeps the tag.
func logosToolTags(tags []string) []string {
	if !Logos {
		return tags
	}
	return append(tags, "logos")
}

// IsLogosFile reports whether name is a loGo/s source file. It is
// mode-independent: it matches ".lgo" and the three alias extensions
// after stripping trailing U+FE0E and U+FE0F variation selectors.
// It reads no mode state, so it may be called at any time.
func IsLogosFile(name string) bool {
	trimmed := strings.TrimRight(name, "\uFE0E\uFE0F")
	if strings.HasSuffix(trimmed, ".lgo") {
		return true
	}
	if strings.HasSuffix(trimmed, ".📜") {
		return true
	}
	if strings.HasSuffix(trimmed, ".⚡") {
		return true
	}
	if strings.HasSuffix(trimmed, ".ʕ◔ϖ◔ʔ") {
		return true
	}
	return false
}

// IsGoFile reports whether name is a Go-family source file: a ".go"
// file, or a loGo/s file when the current process is the logos command.
// It is valid only at run time, after init has set Logos.
func IsGoFile(name string) bool {
	if strings.HasSuffix(name, ".go") {
		return true
	}
	return Logos && IsLogosFile(name)
}

// logosZigArch maps GOARCH to the zig cc -target architecture name.
// logosZigOS maps GOOS to the zig cc -target operating system name;
// the linux entry carries -musl, matching the plan's mapping table.
var logosZigArch = map[string]string{
	"amd64":   "x86_64",
	"arm64":   "aarch64",
	"386":     "x86",
	"arm":     "arm",
	"riscv64": "riscv64",
}

var logosZigOS = map[string]string{
	"linux":   "linux-musl",
	"darwin":  "macos",
	"windows": "windows-gnu",
	"freebsd": "freebsd",
}

// ZigTarget maps a GOOS/GOARCH pair to the zig cc -target triple used by
// the M0-S5 fallback (for example linux/amd64 becomes
// "x86_64-linux-musl"). Only the pairs in the plan's table map; any other
// pair reports false and gets no fallback.
func ZigTarget(goos, goarch string) (string, bool) {
	arch, ok := logosZigArch[goarch]
	if !ok {
		return "", false
	}
	osName, ok := logosZigOS[goos]
	if !ok {
		return "", false
	}
	return arch + "-" + osName, true
}

// StartupCgoEnabled reports the user CGO_ENABLED observed at process
// startup, before main.invoke injects the effective value into the process
// environment. Per-package cgo precedence must use this, not the live
// environment: after invoke the live value always looks explicitly set.
// It returns "" when the user set nothing; only "0" and "1" count as set.
func StartupCgoEnabled() string {
	return logosInitCGOEnabled
}

// RefreshLogosCgoPolicy recomputes the global cgo switch from the
// logos.toml policy of each main module root after -C handling and module
// loading. Callers pass every main module root (all workspace members in
// workspace mode); with no roots the module above the working directory
// decides, and with no module the upstream default stands. A root with an
// explicit cgo key contributes its policy in a loGo/s module and nothing
// in a Go module; without the key an explicitly set CGO_ENABLED decides
// and otherwise a loGo/s module contributes disabled while a Go module
// contributes nothing. Any enabling root turns the build on, so mixed
// workspaces build the allowing modules while per-package checks still
// deny the forbidding ones. A broken logos.toml contributes disabled, so
// the build fails closed. CGOChanged tracks only an explicit user
// CGO_ENABLED, never the file policy. Plain go returns immediately with
// upstream handling untouched. Callers: envcmd MkEnv runs it after the -C
// chdir and before MkEnv bakes CGO_ENABLED into the process environment;
// work BuildInit re-runs it after flag parsing and modload init, where the
// full workspace member list is known.
func RefreshLogosCgoPolicy(mainRoots []string) {
	if !Logos {
		return
	}
	roots := mainRoots
	if len(roots) == 0 {
		if root := cwdModuleRoot(); root != "" {
			roots = []string{root}
		}
	}
	enabled := false
	opinion := false
	for _, root := range roots {
		on, ok := logosRootCgo(root)
		if ok {
			opinion = true
			enabled = enabled || on
		}
	}
	if opinion {
		BuildContext.CgoEnabled = enabled
	}
	if logosInitCGOEnabled == "0" || logosInitCGOEnabled == "1" {
		CGOChanged = true
	} else {
		CGOChanged = false
	}
}

// logosRootCgo reports one main module root's cgo opinion: whether cgo is
// enabled and whether the root says anything at all. Go main modules never
// have an opinion, so they keep the upstream default under logos.
func logosRootCgo(root string) (bool, bool) {
	cf, err := config.Load(root)
	if err != nil || cf == nil {
		return false, true
	}
	if cf.Build.CgoSet {
		if !isLogosModule(root) {
			return false, false
		}
		return cf.Build.Cgo == "allow" || cf.Build.Cgo == "restricted", true
	}
	if logosInitCGOEnabled == "0" || logosInitCGOEnabled == "1" {
		return logosInitCGOEnabled == "1", true
	}
	if !isLogosModule(root) {
		return false, false
	}
	return false, true
}

// LogosCWDModuleRoot reports the module root above the working directory
// by walking up to the nearest go.mod, or "" when there is none. The -O
// level reader uses it to find logos.toml.
func LogosCWDModuleRoot() string {
	return cwdModuleRoot()
}

// cwdModuleRoot returns the module root above the working directory by
// walking up to the nearest go.mod, or "" when there is none. It runs
// after the -C chdir, so -C targets resolve to the target module.
func cwdModuleRoot() string {
	wd, err := os.Getwd()
	if err != nil {
		return ""
	}
	dir := wd
	for {
		if info, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil && !info.IsDir() {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return ""
		}
		dir = parent
	}
}

// isLogosModule reports whether the module rooted at root holds any loGo/s
// source file, so the policy can tell a loGo/s main module (policy
// applies) from a Go main module (upstream handling, cgo key ignored).
// Nested modules decide for themselves: directories holding their own
// go.mod are never descended into. Results are memoized by root.
func isLogosModule(root string) bool {
	if root == "" {
		return false
	}
	logosModuleKindCache.Lock()
	kind, ok := logosModuleKindCache.byRoot[root]
	logosModuleKindCache.Unlock()
	if ok {
		return kind
	}
	kind = logosModuleHasFiles(root)
	logosModuleKindCache.Lock()
	logosModuleKindCache.byRoot[root] = kind
	logosModuleKindCache.Unlock()
	return kind
}

// logosModuleHasFiles walks root for the first loGo/s source file,
// skipping directories that never hold built packages and stopping at
// nested module boundaries.
func logosModuleHasFiles(root string) bool {
	found := false
	filepath.WalkDir(root, func(dir string, d fs.DirEntry, err error) error {
		if found {
			return filepath.SkipAll
		}
		if err != nil {
			return nil
		}
		if !d.IsDir() {
			if IsLogosFile(d.Name()) {
				found = true
				return filepath.SkipAll
			}
			return nil
		}
		if dir == root {
			return nil
		}
		if logosSkippedDir(dir, d) {
			return filepath.SkipDir
		}
		return nil
	})
	return found
}

// logosSkippedDir reports whether dir holds no built packages of root's
// module: dot and underscore prefixes, vendor, testdata, and nested
// modules carrying their own go.mod.
func logosSkippedDir(dir string, d fs.DirEntry) bool {
	name := d.Name()
	if strings.HasPrefix(name, ".") || strings.HasPrefix(name, "_") || name == "vendor" || name == "testdata" {
		return true
	}
	if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
		return true
	}
	return false
}

// zigCompiler resolves one C-family compiler command for a cgo build given
// its current value: the zig fallback "zig <kind> -target <triple>" when
// cgo is on, the variable was unset in both the process environment and
// the GOENV file at startup, the current compiler is missing from PATH,
// and zig is on PATH; otherwise cmd unchanged. kind is "cc" for CC and
// "c++" for CXX. The startup snapshot tells an injected MkEnv default
// (which main.invoke writes into the process environment before builds
// run) from an explicit user setting, so only the former may fall back. It
// never writes the process environment, so "go env CC" and "go env CXX"
// keep showing the default or user compiler. It runs under both go and
// logos for Go and loGo/s packages alike. The triple follows the command
// target, which ForceHost already reset for host tool builds.
func zigCompiler(cmd, startup, kind string) string {
	if !BuildContext.CgoEnabled {
		return cmd
	}
	if startup != "" {
		return cmd
	}
	if _, err := pathcache.LookPath(cmd); err == nil {
		return cmd
	}
	if _, err := pathcache.LookPath("zig"); err != nil {
		return cmd
	}
	if target, ok := ZigTarget(Goos, Goarch); ok {
		return "zig " + kind + " -target " + target
	}
	return cmd
}

// ZigCC resolves the C compiler for a cgo build given the current CC
// value; see zigCompiler. It runs under both go and logos.
func ZigCC(cc string) string {
	return zigCompiler(cc, logosInitCC, "cc")
}

// ZigCXX resolves the C++ compiler for a cgo build given the current CXX
// value; see zigCompiler. It runs under both go and logos.
func ZigCXX(cxx string) string {
	return zigCompiler(cxx, logosInitCXX, "c++")
}

// CgoPolicyAllow reports whether a package may use cgo under the parsed
// logos.toml policy, sharing RefreshLogosCgoPolicy's precedence per
// package: an explicit cgo key wins over CGO_ENABLED both ways (allow
// always, restricted by allowlist, forbid and anything unknown never);
// otherwise an explicit CGO_ENABLED=1 allows and anything else denies
// with the absent-key forbid default. The denial message carries
// [restriction::cgo_usage]; the allowlist message pins the plan's text.
func CgoPolicyAllow(cf *config.File, cgoEnv, importPath string) (bool, string) {
	const deny = "[restriction::cgo_usage] cgo is disabled"
	if cf.Build.CgoSet {
		switch cf.Build.Cgo {
		case "allow":
			return true, ""
		case "restricted":
			if CgoAllowlistAllows(cf.Build.CgoAllowlist, importPath) {
				return true, ""
			}
			return false, fmt.Sprintf("[restriction::cgo_usage] package %s uses cgo and is not in cgo_allowlist", importPath)
		default:
			return false, deny
		}
	}
	if cgoEnv == "1" {
		return true, ""
	}
	return false, deny
}

// CgoAllowlistAllows reports whether importPath matches one of the
// cgo_allowlist globs. A trailing "/**" matches the prefix itself and
// everything under it; a trailing "**" anywhere else matches any suffix
// including path separators; other patterns match exactly or by
// path.Match (*, ?, [...] classes, no recursive wildcard).
func CgoAllowlistAllows(allowlist []string, importPath string) bool {
	for _, pattern := range allowlist {
		if cgoPatternAllows(pattern, importPath) {
			return true
		}
	}
	return false
}

// cgoPatternAllows matches one allowlist glob against one import path per
// SPEC 10.3: a double star matches any suffix.
func cgoPatternAllows(pattern, importPath string) bool {
	if prefix, ok := strings.CutSuffix(pattern, "/**"); ok {
		return importPath == prefix || strings.HasPrefix(importPath, prefix+"/")
	}
	if prefix, ok := strings.CutSuffix(pattern, "**"); ok {
		return strings.HasPrefix(importPath, prefix)
	}
	ok, _ := path.Match(pattern, importPath)
	return ok
}
