// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !cmd_go_bootstrap

package doc

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/load"
	"cmd/go/internal/modload"
	"cmd/go/internal/work"
)

// pickUnusedPort finds an unused port by trying to listen on port 0
// and letting the OS pick a port, then closing that connection and
// returning that port number.
// This is inherently racy.
func pickUnusedPort() (int, error) {
	l, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		return 0, err
	}
	port := l.Addr().(*net.TCPAddr).Port
	if err := l.Close(); err != nil {
		return 0, err
	}
	return port, nil
}

// buildPkgsite builds a pkgsite binary whose build may be cached.
func buildPkgsite(ctx context.Context) string {
	loader := modload.NewLoader()

	// Set the builder to have no module root so we can build a pkg@version pattern.
	loader.ForceUseModules = true
	loader.RootMode = modload.NoRoot
	loader.AllowMissingModuleImports()
	modload.Init(loader)

	work.BuildInit(loader)
	b := work.NewBuilder("", loader.VendorDirOrEmpty)
	defer func() {
		if err := b.Close(); err != nil {
			base.Fatal(err)
		}
	}()

	const version = "v0.0.0-20251223195805-1a3bd3c788fe"
	pkgVers := "golang.org/x/pkgsite/cmd/internal/doc@" + version
	pkgOpts := load.PackageOpts{MainOnly: true}
	pkgs, err := load.PackagesAndErrorsOutsideModule(loader, ctx, pkgOpts, []string{pkgVers})
	if err != nil {
		base.Fatal(err)
	}
	if len(pkgs) == 0 {
		base.Fatalf("go: internal error: no packages loaded for %s", pkgVers)
	}
	if len(pkgs) > 1 {
		base.Fatalf("go: internal error: pattern %s matches multiple packages", pkgVers)
	}
	p := pkgs[0]
	p.Internal.OmitDebug = true
	p.Internal.ExeName = p.DefaultExecName()
	load.CheckPackageErrors([]*load.Package{p})

	a := b.LinkAction(loader, work.ModeBuild, work.ModeBuild, p)
	a.CacheExecutable = true
	b.Do(ctx, a)
	return a.BuiltTarget()
}

func doPkgsite(ctx context.Context, urlPath, fragment string) error {
	port, err := pickUnusedPort()
	if err != nil {
		return fmt.Errorf("failed to find port for documentation server: %v", err)
	}
	addr := fmt.Sprintf("localhost:%d", port)
	path, err := url.JoinPath("http://"+addr, urlPath)
	if err != nil {
		return fmt.Errorf("internal error: failed to construct url: %v", err)
	}
	if fragment != "" {
		path += "#" + fragment
	}

	if file := os.Getenv("TEST_GODOC_URL_FILE"); file != "" {
		return os.WriteFile(file, []byte(path+"\n"), 0666)
	}

	// Turn off the default signal handler for SIGINT (and SIGQUIT on Unix)
	// and instead wait for the child process to handle the signal and
	// exit before exiting ourselves.
	base.StartSigHandlers()

	// Prepend the local download cache to GOPROXY to get around deprecation checks.
	env := os.Environ()
	vars, err := runCmd(env, goCmd(), "env", "GOPROXY", "GOMODCACHE")
	fields := strings.Fields(vars)
	if err == nil && len(fields) == 2 {
		goproxy, gomodcache := fields[0], fields[1]
		gomodcache = filepath.Join(gomodcache, "cache", "download")
		// Convert absolute path to file URL. pkgsite will not accept
		// Windows absolute paths because they look like a host:path remote.
		// TODO(golang.org/issue/32456): use url.FromFilePath when implemented.
		if strings.HasPrefix(gomodcache, "/") {
			gomodcache = "file://" + gomodcache
		} else {
			gomodcache = "file:///" + filepath.ToSlash(gomodcache)
		}
		env = append(env, "GOPROXY="+gomodcache+","+goproxy)
	}

	pkgsite := buildPkgsite(ctx)
	cmd := exec.Command(pkgsite, "-gorepo", cfg.GOROOT, "-http", addr, "-open", path)
	cmd.Env = env
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		if ee, ok := errors.AsType[*exec.ExitError](err); ok {
			// Exit with the same exit status as pkgsite to avoid
			// printing of "exit status" error messages.
			// Any relevant messages have already been printed
			// to stdout or stderr.
			os.Exit(ee.ExitCode())
		}
		return err
	}

	return nil
}
