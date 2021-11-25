// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package work

import (
	"bytes"
	"fmt"
	exec "internal/execabs"
	"os"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/cache"
	"cmd/go/internal/cfg"
	"cmd/go/internal/fsys"
	"cmd/go/internal/str"
	"cmd/internal/buildid"
)

// Build IDs
//
// Go packages and binaries are stamped with build IDs that record both
// the action ID, which is a hash of the inputs to the action that produced
// the packages or binary, and the content ID, which is a hash of the action
// output, namely the archive or binary itself. The hash is the same one
// used by the build artifact cache (see cmd/go/internal/cache), but
// truncated when stored in packages and binaries, as the full length is not
// needed and is a bit unwieldy. The precise form is
//
//	actionID/[.../]contentID
//
// where the actionID and contentID are prepared by buildid.HashToString below.
// and are found by looking for the first or last slash.
// Usually the buildID is simply actionID/contentID, but see below for an
// exception.
//
// The build ID serves two primary purposes.
//
// 1. The action ID half allows installed packages and binaries to serve as
// one-element cache entries. If we intend to build math.a with a given
// set of inputs summarized in the action ID, and the installed math.a already
// has that action ID, we can reuse the installed math.a instead of rebuilding it.
//
// 2. The content ID half allows the easy preparation of action IDs for steps
// that consume a particular package or binary. The content hash of every
// input file for a given action must be included in the action ID hash.
// Storing the content ID in the build ID lets us read it from the file with
// minimal I/O, instead of reading and hashing the entire file.
// This is especially effective since packages and binaries are typically
// the largest inputs to an action.
//
// Separating action ID from content ID is important for reproducible builds.
// The compiler is compiled with itself. If an output were represented by its
// own action ID (instead of content ID) when computing the action ID of
// the next step in the build process, then the compiler could never have its
// own input action ID as its output action ID (short of a miraculous hash collision).
// Instead we use the content IDs to compute the next action ID, and because
// the content IDs converge, so too do the action IDs and therefore the
// build IDs and the overall compiler binary. See cmd/dist's cmdbootstrap
// for the actual convergence sequence.
//
// The “one-element cache” purpose is a bit more complex for installed
// binaries. For a binary, like cmd/gofmt, there are two steps: compile
// cmd/gofmt/*.go into main.a, and then link main.a into the gofmt binary.
// We do not install gofmt's main.a, only the gofmt binary. Being able to
// decide that the gofmt binary is up-to-date means computing the action ID
// for the final link of the gofmt binary and comparing it against the
// already-installed gofmt binary. But computing the action ID for the link
// means knowing the content ID of main.a, which we did not keep.
// To sidestep this problem, each binary actually stores an expanded build ID:
//
//	actionID(binary)/actionID(main.a)/contentID(main.a)/contentID(binary)
//
// (Note that this can be viewed equivalently as:
//
//	actionID(binary)/buildID(main.a)/contentID(binary)
//
// Storing the buildID(main.a) in the middle lets the computations that care
// about the prefix or suffix halves ignore the middle and preserves the
// original build ID as a contiguous string.)
//
// During the build, when it's time to build main.a, the gofmt binary has the
// information needed to decide whether the eventual link would produce
// the same binary: if the action ID for main.a's inputs matches and then
// the action ID for the link step matches when assuming the given main.a
// content ID, then the binary as a whole is up-to-date and need not be rebuilt.
//
// This is all a bit complex and may be simplified once we can rely on the
// main cache, but at least at the start we will be using the content-based
// staleness determination without a cache beyond the usual installed
// package and binary locations.

const buildIDSeparator = "/"

// actionID returns the action ID half of a build ID.
func actionID(buildID string) string {
	i := strings.Index(buildID, buildIDSeparator)
	if i < 0 {
		return buildID
	}
	return buildID[:i]
}

// contentID returns the content ID half of a build ID.
func contentID(buildID string) string {
	return buildID[strings.LastIndex(buildID, buildIDSeparator)+1:]
}

// toolID returns the unique ID to use for the current copy of the
// named tool (asm, compile, cover, link).
//
// It is important that if the tool changes (for example a compiler bug is fixed
// and the compiler reinstalled), toolID returns a different string, so that old
// package archives look stale and are rebuilt (with the fixed compiler).
// This suggests using a content hash of the tool binary, as stored in the build ID.
//
// Unfortunately, we can't just open the tool binary, because the tool might be
// invoked via a wrapper program specified by -toolexec and we don't know
// what the wrapper program does. In particular, we want "-toolexec toolstash"
// to continue working: it does no good if "-toolexec toolstash" is executing a
// stashed copy of the compiler but the go command is acting as if it will run
// the standard copy of the compiler. The solution is to ask the tool binary to tell
// us its own build ID using the "-V=full" flag now supported by all tools.
// Then we know we're getting the build ID of the compiler that will actually run
// during the build. (How does the compiler binary know its own content hash?
// We store it there using updateBuildID after the standard link step.)
//
// A final twist is that we'd prefer to have reproducible builds for release toolchains.
// It should be possible to cross-compile for Windows from either Linux or Mac
// or Windows itself and produce the same binaries, bit for bit. If the tool ID,
// which influences the action ID half of the build ID, is based on the content ID,
// then the Linux compiler binary and Mac compiler binary will have different tool IDs
// and therefore produce executables with different action IDs.
// To avoid this problem, for releases we use the release version string instead
// of the compiler binary's content hash. This assumes that all compilers built
// on all different systems are semantically equivalent, which is of course only true
// modulo bugs. (Producing the exact same executables also requires that the different
// build setups agree on details like $GOROOT and file name paths, but at least the
// tool IDs do not make it impossible.)
func (b *Builder) toolID(name string) string {
	b.id.Lock()
	id := b.toolIDCache[name]
	b.id.Unlock()

	if id != "" {
		return id
	}

	path := base.Tool(name)
	desc := "go tool " + name

	// Special case: undocumented -vettool overrides usual vet,
	// for testing vet or supplying an alternative analysis tool.
	if name == "vet" && VetTool != "" {
		path = VetTool
		desc = VetTool
	}

	cmdline := str.StringList(cfg.BuildToolexec, path, "-V=full")
	cmd := exec.Command(cmdline[0], cmdline[1:]...)
	cmd.Env = base.AppendPWD(os.Environ(), cmd.Dir)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		base.Fatalf("%s: %v\n%s%s", desc, err, stdout.Bytes(), stderr.Bytes())
	}

	line := stdout.String()
	f := strings.Fields(line)
	if len(f) < 3 || f[0] != name && path != VetTool || f[1] != "version" || f[2] == "devel" && !strings.HasPrefix(f[len(f)-1], "buildID=") {
		base.Fatalf("%s -V=full: unexpected output:\n\t%s", desc, line)
	}
	if f[2] == "devel" {
		// On the development branch, use the content ID part of the build ID.
		id = contentID(f[len(f)-1])
	} else {
		// For a release, the output is like: "compile version go1.9.1 X:framepointer".
		// Use the whole line.
		id = strings.TrimSpace(line)
	}

	b.id.Lock()
	b.toolIDCache[name] = id
	b.id.Unlock()

	return id
}

// gccToolID returns the unique ID to use for a tool that is invoked
// by the GCC driver. This is used particularly for gccgo, but this can also
// be used for gcc, g++, gfortran, etc.; those tools all use the GCC
// driver under different names. The approach used here should also
// work for sufficiently new versions of clang. Unlike toolID, the
// name argument is the program to run. The language argument is the
// type of input file as passed to the GCC driver's -x option.
//
// For these tools we have no -V=full option to dump the build ID,
// but we can run the tool with -v -### to reliably get the compiler proper
// and hash that. That will work in the presence of -toolexec.
//
// In order to get reproducible builds for released compilers, we
// detect a released compiler by the absence of "experimental" in the
// --version output, and in that case we just use the version string.
func (b *Builder) gccToolID(name, language string) (string, error) {
	key := name + "." + language
	b.id.Lock()
	id := b.toolIDCache[key]
	b.id.Unlock()

	if id != "" {
		return id, nil
	}

	// Invoke the driver with -### to see the subcommands and the
	// version strings. Use -x to set the language. Pretend to
	// compile an empty file on standard input.
	cmdline := str.StringList(cfg.BuildToolexec, name, "-###", "-x", language, "-c", "-")
	cmd := exec.Command(cmdline[0], cmdline[1:]...)
	cmd.Env = base.AppendPWD(os.Environ(), cmd.Dir)
	// Force untranslated output so that we see the string "version".
	cmd.Env = append(cmd.Env, "LC_ALL=C")
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("%s: %v; output: %q", name, err, out)
	}

	version := ""
	lines := strings.Split(string(out), "\n")
	for _, line := range lines {
		if fields := strings.Fields(line); len(fields) > 1 && fields[1] == "version" {
			version = line
			break
		}
	}
	if version == "" {
		return "", fmt.Errorf("%s: can not find version number in %q", name, out)
	}

	if !strings.Contains(version, "experimental") {
		// This is a release. Use this line as the tool ID.
		id = version
	} else {
		// This is a development version. The first line with
		// a leading space is the compiler proper.
		compiler := ""
		for _, line := range lines {
			if len(line) > 1 && line[0] == ' ' {
				compiler = line
				break
			}
		}
		if compiler == "" {
			return "", fmt.Errorf("%s: can not find compilation command in %q", name, out)
		}

		fields := strings.Fields(compiler)
		if len(fields) == 0 {
			return "", fmt.Errorf("%s: compilation command confusion %q", name, out)
		}
		exe := fields[0]
		if !strings.ContainsAny(exe, `/\`) {
			if lp, err := exec.LookPath(exe); err == nil {
				exe = lp
			}
		}
		id, err = buildid.ReadFile(exe)
		if err != nil {
			return "", err
		}

		// If we can't find a build ID, use a hash.
		if id == "" {
			id = b.fileHash(exe)
		}
	}

	b.id.Lock()
	b.toolIDCache[key] = id
	b.id.Unlock()

	return id, nil
}

// Check if assembler used by gccgo is GNU as.
func assemblerIsGas() bool {
	cmd := exec.Command(BuildToolchain.compiler(), "-print-prog-name=as")
	assembler, err := cmd.Output()
	if err == nil {
		cmd := exec.Command(strings.TrimSpace(string(assembler)), "--version")
		out, err := cmd.Output()
		return err == nil && strings.Contains(string(out), "GNU")
	} else {
		return false
	}
}

// gccgoBuildIDFile creates an assembler file that records the
// action's build ID in an SHF_EXCLUDE section for ELF files or
// in a CSECT in XCOFF files.
func (b *Builder) gccgoBuildIDFile(a *Action) (string, error) {
	sfile := a.Objdir + "_buildid.s"

	var buf bytes.Buffer
	if cfg.Goos == "aix" {
		fmt.Fprintf(&buf, "\t.csect .go.buildid[XO]\n")
	} else if (cfg.Goos != "solaris" && cfg.Goos != "illumos") || assemblerIsGas() {
		fmt.Fprintf(&buf, "\t"+`.section .go.buildid,"e"`+"\n")
	} else if cfg.Goarch == "sparc" || cfg.Goarch == "sparc64" {
		fmt.Fprintf(&buf, "\t"+`.section ".go.buildid",#exclude`+"\n")
	} else { // cfg.Goarch == "386" || cfg.Goarch == "amd64"
		fmt.Fprintf(&buf, "\t"+`.section .go.buildid,#exclude`+"\n")
	}
	fmt.Fprintf(&buf, "\t.byte ")
	for i := 0; i < len(a.buildID); i++ {
		if i > 0 {
			if i%8 == 0 {
				fmt.Fprintf(&buf, "\n\t.byte ")
			} else {
				fmt.Fprintf(&buf, ",")
			}
		}
		fmt.Fprintf(&buf, "%#02x", a.buildID[i])
	}
	fmt.Fprintf(&buf, "\n")
	if cfg.Goos != "solaris" && cfg.Goos != "illumos" && cfg.Goos != "aix" {
		secType := "@progbits"
		if cfg.Goarch == "arm" {
			secType = "%progbits"
		}
		fmt.Fprintf(&buf, "\t"+`.section .note.GNU-stack,"",%s`+"\n", secType)
		fmt.Fprintf(&buf, "\t"+`.section .note.GNU-split-stack,"",%s`+"\n", secType)
	}

	if cfg.BuildN || cfg.BuildX {
		for _, line := range bytes.Split(buf.Bytes(), []byte("\n")) {
			b.Showcmd("", "echo '%s' >> %s", line, sfile)
		}
		if cfg.BuildN {
			return sfile, nil
		}
	}

	if err := os.WriteFile(sfile, buf.Bytes(), 0666); err != nil {
		return "", err
	}

	return sfile, nil
}

// buildID returns the build ID found in the given file.
// If no build ID is found, buildID returns the content hash of the file.
func (b *Builder) buildID(file string) string {
	b.id.Lock()
	id := b.buildIDCache[file]
	b.id.Unlock()

	if id != "" {
		return id
	}

	id, err := buildid.ReadFile(file)
	if err != nil {
		id = b.fileHash(file)
	}

	b.id.Lock()
	b.buildIDCache[file] = id
	b.id.Unlock()

	return id
}

// fileHash returns the content hash of the named file.
func (b *Builder) fileHash(file string) string {
	file, _ = fsys.OverlayPath(file)
	sum, err := cache.FileHash(file)
	if err != nil {
		return ""
	}
	return buildid.HashToString(sum)
}

// useCache tries to satisfy the action a, which has action ID actionHash,
// by using a cached result from an earlier build. At the moment, the only
// cached result is the installed package or binary at target.
// If useCache decides that the cache can be used, it sets a.buildID
// and a.built for use by parent actions and then returns true.
// Otherwise it sets a.buildID to a temporary build ID for use in the build
// and returns false. When useCache returns false the expectation is that
// the caller will build the target and then call updateBuildID to finish the
// build ID computation.
// When useCache returns false, it may have initiated buffering of output
// during a's work. The caller should defer b.flushOutput(a), to make sure
// that flushOutput is eventually called regardless of whether the action
// succeeds. The flushOutput call must happen after updateBuildID.
func (b *Builder) useCache(a *Action, actionHash cache.ActionID, target string) bool {
	// The second half of the build ID here is a placeholder for the content hash.
	// It's important that the overall buildID be unlikely verging on impossible
	// to appear in the output by chance, but that should be taken care of by
	// the actionID half; if it also appeared in the input that would be like an
	// engineered 120-bit partial SHA256 collision.
	a.actionID = actionHash
	actionID := buildid.HashToString(actionHash)
	if a.json != nil {
		a.json.ActionID = actionID
	}
	contentID := actionID // temporary placeholder, likely unique
	a.buildID = actionID + buildIDSeparator + contentID

	// Executable binaries also record the main build ID in the middle.
	// See "Build IDs" comment above.
	if a.Mode == "link" {
		mainpkg := a.Deps[0]
		a.buildID = actionID + buildIDSeparator + mainpkg.buildID + buildIDSeparator + contentID
	}

	// Check to see if target exists and matches the expected action ID.
	// If so, it's up to date and we can reuse it instead of rebuilding it.
	var buildID string
	if target != "" && !cfg.BuildA {
		buildID, _ = buildid.ReadFile(target)
		if strings.HasPrefix(buildID, actionID+buildIDSeparator) {
			a.buildID = buildID
			if a.json != nil {
				a.json.BuildID = a.buildID
			}
			a.built = target
			// Poison a.Target to catch uses later in the build.
			a.Target = "DO NOT USE - " + a.Mode
			return true
		}
	}

	// Special case for building a main package: if the only thing we
	// want the package for is to link a binary, and the binary is
	// already up-to-date, then to avoid a rebuild, report the package
	// as up-to-date as well. See "Build IDs" comment above.
	// TODO(rsc): Rewrite this code to use a TryCache func on the link action.
	if target != "" && !cfg.BuildA && !b.NeedExport && a.Mode == "build" && len(a.triggers) == 1 && a.triggers[0].Mode == "link" {
		buildID, err := buildid.ReadFile(target)
		if err == nil {
			id := strings.Split(buildID, buildIDSeparator)
			if len(id) == 4 && id[1] == actionID {
				// Temporarily assume a.buildID is the package build ID
				// stored in the installed binary, and see if that makes
				// the upcoming link action ID a match. If so, report that
				// we built the package, safe in the knowledge that the
				// link step will not ask us for the actual package file.
				// Note that (*Builder).LinkAction arranged that all of
				// a.triggers[0]'s dependencies other than a are also
				// dependencies of a, so that we can be sure that,
				// other than a.buildID, b.linkActionID is only accessing
				// build IDs of completed actions.
				oldBuildID := a.buildID
				a.buildID = id[1] + buildIDSeparator + id[2]
				linkID := buildid.HashToString(b.linkActionID(a.triggers[0]))
				if id[0] == linkID {
					// Best effort attempt to display output from the compile and link steps.
					// If it doesn't work, it doesn't work: reusing the cached binary is more
					// important than reprinting diagnostic information.
					if c := cache.Default(); c != nil {
						showStdout(b, c, a.actionID, "stdout")      // compile output
						showStdout(b, c, a.actionID, "link-stdout") // link output
					}

					// Poison a.Target to catch uses later in the build.
					a.Target = "DO NOT USE - main build pseudo-cache Target"
					a.built = "DO NOT USE - main build pseudo-cache built"
					if a.json != nil {
						a.json.BuildID = a.buildID
					}
					return true
				}
				// Otherwise restore old build ID for main build.
				a.buildID = oldBuildID
			}
		}
	}

	// Special case for linking a test binary: if the only thing we
	// want the binary for is to run the test, and the test result is cached,
	// then to avoid the link step, report the link as up-to-date.
	// We avoid the nested build ID problem in the previous special case
	// by recording the test results in the cache under the action ID half.
	if !cfg.BuildA && len(a.triggers) == 1 && a.triggers[0].TryCache != nil && a.triggers[0].TryCache(b, a.triggers[0]) {
		// Best effort attempt to display output from the compile and link steps.
		// If it doesn't work, it doesn't work: reusing the test result is more
		// important than reprinting diagnostic information.
		if c := cache.Default(); c != nil {
			showStdout(b, c, a.Deps[0].actionID, "stdout")      // compile output
			showStdout(b, c, a.Deps[0].actionID, "link-stdout") // link output
		}

		// Poison a.Target to catch uses later in the build.
		a.Target = "DO NOT USE -  pseudo-cache Target"
		a.built = "DO NOT USE - pseudo-cache built"
		return true
	}

	if b.IsCmdList {
		// Invoked during go list to compute and record staleness.
		if p := a.Package; p != nil && !p.Stale {
			p.Stale = true
			if cfg.BuildA {
				p.StaleReason = "build -a flag in use"
			} else {
				p.StaleReason = "build ID mismatch"
				for _, p1 := range p.Internal.Imports {
					if p1.Stale && p1.StaleReason != "" {
						if strings.HasPrefix(p1.StaleReason, "stale dependency: ") {
							p.StaleReason = p1.StaleReason
							break
						}
						if strings.HasPrefix(p.StaleReason, "build ID mismatch") {
							p.StaleReason = "stale dependency: " + p1.ImportPath
						}
					}
				}
			}
		}

		// Fall through to update a.buildID from the build artifact cache,
		// which will affect the computation of buildIDs for targets
		// higher up in the dependency graph.
	}

	// Check the build artifact cache.
	// We treat hits in this cache as being "stale" for the purposes of go list
	// (in effect, "stale" means whether p.Target is up-to-date),
	// but we're still happy to use results from the build artifact cache.
	if c := cache.Default(); c != nil {
		if !cfg.BuildA {
			if file, _, err := c.GetFile(actionHash); err == nil {
				if buildID, err := buildid.ReadFile(file); err == nil {
					if err := showStdout(b, c, a.actionID, "stdout"); err == nil {
						a.built = file
						a.Target = "DO NOT USE - using cache"
						a.buildID = buildID
						if a.json != nil {
							a.json.BuildID = a.buildID
						}
						if p := a.Package; p != nil {
							// Clearer than explaining that something else is stale.
							p.StaleReason = "not installed but available in build cache"
						}
						return true
					}
				}
			}
		}

		// Begin saving output for later writing to cache.
		a.output = []byte{}
	}

	return false
}

func showStdout(b *Builder, c *cache.Cache, actionID cache.ActionID, key string) error {
	stdout, stdoutEntry, err := c.GetBytes(cache.Subkey(actionID, key))
	if err != nil {
		return err
	}

	if len(stdout) > 0 {
		if cfg.BuildX || cfg.BuildN {
			b.Showcmd("", "%s  # internal", joinUnambiguously(str.StringList("cat", c.OutputFile(stdoutEntry.OutputID))))
		}
		if !cfg.BuildN {
			b.Print(string(stdout))
		}
	}
	return nil
}

// flushOutput flushes the output being queued in a.
func (b *Builder) flushOutput(a *Action) {
	b.Print(string(a.output))
	a.output = nil
}

// updateBuildID updates the build ID in the target written by action a.
// It requires that useCache was called for action a and returned false,
// and that the build was then carried out and given the temporary
// a.buildID to record as the build ID in the resulting package or binary.
// updateBuildID computes the final content ID and updates the build IDs
// in the binary.
//
// Keep in sync with src/cmd/buildid/buildid.go
func (b *Builder) updateBuildID(a *Action, target string, rewrite bool) error {
	if cfg.BuildX || cfg.BuildN {
		if rewrite {
			b.Showcmd("", "%s # internal", joinUnambiguously(str.StringList(base.Tool("buildid"), "-w", target)))
		}
		if cfg.BuildN {
			return nil
		}
	}

	// Cache output from compile/link, even if we don't do the rest.
	if c := cache.Default(); c != nil {
		switch a.Mode {
		case "build":
			c.PutBytes(cache.Subkey(a.actionID, "stdout"), a.output)
		case "link":
			// Even though we don't cache the binary, cache the linker text output.
			// We might notice that an installed binary is up-to-date but still
			// want to pretend to have run the linker.
			// Store it under the main package's action ID
			// to make it easier to find when that's all we have.
			for _, a1 := range a.Deps {
				if p1 := a1.Package; p1 != nil && p1.Name == "main" {
					c.PutBytes(cache.Subkey(a1.actionID, "link-stdout"), a.output)
					break
				}
			}
		}
	}

	// Find occurrences of old ID and compute new content-based ID.
	r, err := os.Open(target)
	if err != nil {
		return err
	}
	matches, hash, err := buildid.FindAndHash(r, a.buildID, 0)
	r.Close()
	if err != nil {
		return err
	}
	newID := a.buildID[:strings.LastIndex(a.buildID, buildIDSeparator)] + buildIDSeparator + buildid.HashToString(hash)
	if len(newID) != len(a.buildID) {
		return fmt.Errorf("internal error: build ID length mismatch %q vs %q", a.buildID, newID)
	}

	// Replace with new content-based ID.
	a.buildID = newID
	if a.json != nil {
		a.json.BuildID = a.buildID
	}
	if len(matches) == 0 {
		// Assume the user specified -buildid= to override what we were going to choose.
		return nil
	}

	if rewrite {
		w, err := os.OpenFile(target, os.O_RDWR, 0)
		if err != nil {
			return err
		}
		err = buildid.Rewrite(w, matches, newID)
		if err != nil {
			w.Close()
			return err
		}
		if err := w.Close(); err != nil {
			return err
		}
	}

	// Cache package builds, but not binaries (link steps).
	// The expectation is that binaries are not reused
	// nearly as often as individual packages, and they're
	// much larger, so the cache-footprint-to-utility ratio
	// of binaries is much lower for binaries.
	// Not caching the link step also makes sure that repeated "go run" at least
	// always rerun the linker, so that they don't get too fast.
	// (We don't want people thinking go is a scripting language.)
	// Note also that if we start caching binaries, then we will
	// copy the binaries out of the cache to run them, and then
	// that will mean the go process is itself writing a binary
	// and then executing it, so we will need to defend against
	// ETXTBSY problems as discussed in exec.go and golang.org/issue/22220.
	if c := cache.Default(); c != nil && a.Mode == "build" {
		r, err := os.Open(target)
		if err == nil {
			if a.output == nil {
				panic("internal error: a.output not set")
			}
			outputID, _, err := c.Put(a.actionID, r)
			r.Close()
			if err == nil && cfg.BuildX {
				b.Showcmd("", "%s # internal", joinUnambiguously(str.StringList("cp", target, c.OutputFile(outputID))))
			}
			if b.NeedExport {
				if err != nil {
					return err
				}
				a.Package.Export = c.OutputFile(outputID)
				a.Package.BuildID = a.buildID
			}
		}
	}

	return nil
}
