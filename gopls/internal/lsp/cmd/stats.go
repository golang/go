// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmd

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"time"

	goplsbug "golang.org/x/tools/gopls/internal/bug"
	"golang.org/x/tools/gopls/internal/lsp"
	"golang.org/x/tools/gopls/internal/lsp/command"
	"golang.org/x/tools/gopls/internal/lsp/debug"
	"golang.org/x/tools/gopls/internal/lsp/filecache"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/internal/event"
)

type stats struct {
	app *Application

	Anon bool `flag:"anon" help:"hide any fields that may contain user names, file names, or source code"`
}

func (s *stats) Name() string      { return "stats" }
func (r *stats) Parent() string    { return r.app.Name() }
func (s *stats) Usage() string     { return "" }
func (s *stats) ShortHelp() string { return "print workspace statistics" }

func (s *stats) DetailedHelp(f *flag.FlagSet) {
	fmt.Fprint(f.Output(), `
Load the workspace for the current directory, and output a JSON summary of
workspace information relevant to performance. As a side effect, this command
populates the gopls file cache for the current workspace.

By default, this command may include output that refers to the location or
content of user code. When the -anon flag is set, fields that may refer to user
code are hidden.

Example:
  $ gopls stats -anon
`)
	printFlagDefaults(f)
}

func (s *stats) Run(ctx context.Context, args ...string) error {
	// This undocumented environment variable allows
	// the cmd integration test to trigger a call to bug.Report.
	if msg := os.Getenv("TEST_GOPLS_BUG"); msg != "" {
		filecache.Start() // effect: register bug handler
		goplsbug.Report(msg)
		return nil
	}

	if s.app.Remote != "" {
		// stats does not work with -remote.
		// Other sessions on the daemon may interfere with results.
		// Additionally, the type assertions in below only work if progress
		// notifications bypass jsonrpc2 serialization.
		return fmt.Errorf("the stats subcommand does not work with -remote")
	}

	if !s.app.Verbose {
		event.SetExporter(nil) // don't log errors to stderr
	}

	stats := GoplsStats{
		GOOS:         runtime.GOOS,
		GOARCH:       runtime.GOARCH,
		GOPLSCACHE:   os.Getenv("GOPLSCACHE"),
		GoVersion:    runtime.Version(),
		GoplsVersion: debug.Version,
	}

	opts := s.app.options
	s.app.options = func(o *source.Options) {
		if opts != nil {
			opts(o)
		}
		o.VerboseWorkDoneProgress = true
	}
	var (
		iwlMu    sync.Mutex
		iwlToken protocol.ProgressToken
		iwlDone  = make(chan struct{})
	)

	onProgress := func(p *protocol.ProgressParams) {
		switch v := p.Value.(type) {
		case *protocol.WorkDoneProgressBegin:
			if v.Title == lsp.DiagnosticWorkTitle(lsp.FromInitialWorkspaceLoad) {
				iwlMu.Lock()
				iwlToken = p.Token
				iwlMu.Unlock()
			}
		case *protocol.WorkDoneProgressEnd:
			iwlMu.Lock()
			tok := iwlToken
			iwlMu.Unlock()

			if p.Token == tok {
				close(iwlDone)
			}
		}
	}

	// do executes a timed section of the stats command.
	do := func(name string, f func() error) (time.Duration, error) {
		start := time.Now()
		fmt.Fprintf(os.Stderr, "%-30s", name+"...")
		if err := f(); err != nil {
			return time.Since(start), err
		}
		d := time.Since(start)
		fmt.Fprintf(os.Stderr, "done (%v)\n", d)
		return d, nil
	}

	var conn *connection
	iwlDuration, err := do("Initializing workspace", func() error {
		var err error
		conn, err = s.app.connect(ctx, onProgress)
		if err != nil {
			return err
		}
		select {
		case <-iwlDone:
		case <-ctx.Done():
			return ctx.Err()
		}
		return nil
	})
	stats.InitialWorkspaceLoadDuration = fmt.Sprint(iwlDuration)
	if err != nil {
		return err
	}
	defer conn.terminate(ctx)

	// Gather bug reports produced by any process using
	// this executable and persisted in the cache.
	do("Gathering bug reports", func() error {
		stats.CacheDir, stats.BugReports = filecache.BugReports()
		if stats.BugReports == nil {
			stats.BugReports = []goplsbug.Bug{} // non-nil for JSON
		}
		return nil
	})

	if _, err := do("Querying memstats", func() error {
		memStats, err := conn.ExecuteCommand(ctx, &protocol.ExecuteCommandParams{
			Command: command.MemStats.ID(),
		})
		if err != nil {
			return err
		}
		stats.MemStats = memStats.(command.MemStatsResult)
		return nil
	}); err != nil {
		return err
	}

	if _, err := do("Querying workspace stats", func() error {
		wsStats, err := conn.ExecuteCommand(ctx, &protocol.ExecuteCommandParams{
			Command: command.WorkspaceStats.ID(),
		})
		if err != nil {
			return err
		}
		stats.WorkspaceStats = wsStats.(command.WorkspaceStatsResult)
		return nil
	}); err != nil {
		return err
	}

	if _, err := do("Collecting directory info", func() error {
		var err error
		stats.DirStats, err = findDirStats(ctx)
		if err != nil {
			return err
		}
		return nil
	}); err != nil {
		return err
	}

	// Filter JSON output to fields that are consistent with s.Anon.
	okFields := make(map[string]interface{})
	{
		v := reflect.ValueOf(stats)
		t := v.Type()
		for i := 0; i < t.NumField(); i++ {
			f := t.Field(i)
			if !token.IsExported(f.Name) {
				continue
			}
			if s.Anon && f.Tag.Get("anon") != "ok" {
				// Fields that can be served with -anon must be explicitly marked as OK.
				continue
			}
			vf := v.FieldByName(f.Name)
			okFields[f.Name] = vf.Interface()
		}
	}
	data, err := json.MarshalIndent(okFields, "", "  ")
	if err != nil {
		return err
	}

	os.Stdout.Write(data)
	fmt.Println()
	return nil
}

// GoplsStats holds information extracted from a gopls session in the current
// workspace.
//
// Fields that should be printed with the -anon flag should be explicitly
// marked as `anon:"ok"`. Only fields that cannot refer to user files or code
// should be marked as such.
type GoplsStats struct {
	GOOS, GOARCH                 string `anon:"ok"`
	GOPLSCACHE                   string
	GoVersion                    string `anon:"ok"`
	GoplsVersion                 string `anon:"ok"`
	InitialWorkspaceLoadDuration string `anon:"ok"` // in time.Duration string form
	CacheDir                     string
	BugReports                   []goplsbug.Bug
	MemStats                     command.MemStatsResult       `anon:"ok"`
	WorkspaceStats               command.WorkspaceStatsResult `anon:"ok"`
	DirStats                     dirStats                     `anon:"ok"`
}

type dirStats struct {
	Files         int
	TestdataFiles int
	GoFiles       int
	ModFiles      int
	Dirs          int
}

// findDirStats collects information about the current directory and its
// subdirectories.
func findDirStats(ctx context.Context) (dirStats, error) {
	var ds dirStats
	filepath.WalkDir(".", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			ds.Dirs++
		} else {
			ds.Files++
			slashed := filepath.ToSlash(path)
			switch {
			case strings.Contains(slashed, "/testdata/") || strings.HasPrefix(slashed, "testdata/"):
				ds.TestdataFiles++
			case strings.HasSuffix(path, ".go"):
				ds.GoFiles++
			case strings.HasSuffix(path, ".mod"):
				ds.ModFiles++
			}
		}
		return nil
	})
	return ds, nil
}
