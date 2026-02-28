// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modload

import (
	"context"
	"flag"
	"internal/testenv"
	"log"
	"os"
	"path"
	"path/filepath"
	"strings"
	"testing"

	"cmd/go/internal/cfg"
	"cmd/go/internal/vcweb/vcstest"

	"golang.org/x/mod/module"
)

func TestMain(m *testing.M) {
	flag.Parse()
	if err := testMain(m); err != nil {
		log.Fatal(err)
	}
}

func testMain(m *testing.M) (err error) {
	cfg.GOPROXY = "direct"
	cfg.ModCacheRW = true

	srv, err := vcstest.NewServer()
	if err != nil {
		return err
	}
	defer func() {
		if closeErr := srv.Close(); err == nil {
			err = closeErr
		}
	}()

	dir, err := os.MkdirTemp("", "modload-test-")
	if err != nil {
		return err
	}
	defer func() {
		if rmErr := os.RemoveAll(dir); err == nil {
			err = rmErr
		}
	}()

	os.Setenv("GOPATH", dir)
	cfg.BuildContext.GOPATH = dir
	cfg.GOMODCACHE = filepath.Join(dir, "pkg/mod")
	cfg.SumdbDir = filepath.Join(dir, "pkg/sumdb")
	m.Run()
	return nil
}

var (
	queryRepo   = "vcs-test.golang.org/git/querytest.git"
	queryRepoV2 = queryRepo + "/v2"
	queryRepoV3 = queryRepo + "/v3"

	// Empty version list (no semver tags), not actually empty.
	emptyRepoPath = "vcs-test.golang.org/git/emptytest.git"
)

var queryTests = []struct {
	path    string
	query   string
	current string
	allow   string
	vers    string
	err     string
}{
	{path: queryRepo, query: "<v0.0.0", vers: "v0.0.0-pre1"},
	{path: queryRepo, query: "<v0.0.0-pre1", err: `no matching versions for query "<v0.0.0-pre1"`},
	{path: queryRepo, query: "<=v0.0.0", vers: "v0.0.0"},
	{path: queryRepo, query: ">v0.0.0", vers: "v0.0.1"},
	{path: queryRepo, query: ">=v0.0.0", vers: "v0.0.0"},
	{path: queryRepo, query: "v0.0.1", vers: "v0.0.1"},
	{path: queryRepo, query: "v0.0.1+foo", vers: "v0.0.1"},
	{path: queryRepo, query: "v0.0.99", err: `vcs-test.golang.org/git/querytest.git@v0.0.99: invalid version: unknown revision v0.0.99`},
	{path: queryRepo, query: "v0", vers: "v0.3.0"},
	{path: queryRepo, query: "v0.1", vers: "v0.1.2"},
	{path: queryRepo, query: "v0.2", err: `no matching versions for query "v0.2"`},
	{path: queryRepo, query: "v0.0", vers: "v0.0.3"},
	{path: queryRepo, query: "v1.9.10-pre2+metadata", vers: "v1.9.10-pre2.0.20190513201126-42abcb6df8ee"},
	{path: queryRepo, query: "ed5ffdaa", vers: "v1.9.10-pre2.0.20191220134614-ed5ffdaa1f5e"},

	// golang.org/issue/29262: The major version for a module without a suffix
	// should be based on the most recent tag (v1 as appropriate, not v0
	// unconditionally).
	{path: queryRepo, query: "42abcb6df8ee", vers: "v1.9.10-pre2.0.20190513201126-42abcb6df8ee"},

	{path: queryRepo, query: "v1.9.10-pre2+wrongmetadata", err: `vcs-test.golang.org/git/querytest.git@v1.9.10-pre2+wrongmetadata: invalid version: unknown revision v1.9.10-pre2+wrongmetadata`},
	{path: queryRepo, query: "v1.9.10-pre2", err: `vcs-test.golang.org/git/querytest.git@v1.9.10-pre2: invalid version: unknown revision v1.9.10-pre2`},
	{path: queryRepo, query: "latest", vers: "v1.9.9"},
	{path: queryRepo, query: "latest", current: "v1.9.10-pre1", vers: "v1.9.9"},
	{path: queryRepo, query: "upgrade", vers: "v1.9.9"},
	{path: queryRepo, query: "upgrade", current: "v1.9.10-pre1", vers: "v1.9.10-pre1"},
	{path: queryRepo, query: "upgrade", current: "v1.9.10-pre2+metadata", vers: "v1.9.10-pre2.0.20190513201126-42abcb6df8ee"},
	{path: queryRepo, query: "upgrade", current: "v0.0.0-20190513201126-42abcb6df8ee", vers: "v0.0.0-20190513201126-42abcb6df8ee"},
	{path: queryRepo, query: "upgrade", allow: "NOMATCH", err: `no matching versions for query "upgrade"`},
	{path: queryRepo, query: "upgrade", current: "v1.9.9", allow: "NOMATCH", err: `vcs-test.golang.org/git/querytest.git@v1.9.9: disallowed module version`},
	{path: queryRepo, query: "upgrade", current: "v1.99.99", err: `vcs-test.golang.org/git/querytest.git@v1.99.99: invalid version: unknown revision v1.99.99`},
	{path: queryRepo, query: "patch", current: "", err: `can't query version "patch" of module vcs-test.golang.org/git/querytest.git: no existing version is required`},
	{path: queryRepo, query: "patch", current: "v0.1.0", vers: "v0.1.2"},
	{path: queryRepo, query: "patch", current: "v1.9.0", vers: "v1.9.9"},
	{path: queryRepo, query: "patch", current: "v1.9.10-pre1", vers: "v1.9.10-pre1"},
	{path: queryRepo, query: "patch", current: "v1.9.10-pre2+metadata", vers: "v1.9.10-pre2.0.20190513201126-42abcb6df8ee"},
	{path: queryRepo, query: "patch", current: "v1.99.99", err: `vcs-test.golang.org/git/querytest.git@v1.99.99: invalid version: unknown revision v1.99.99`},
	{path: queryRepo, query: ">v1.9.9", vers: "v1.9.10-pre1"},
	{path: queryRepo, query: ">v1.10.0", err: `no matching versions for query ">v1.10.0"`},
	{path: queryRepo, query: ">=v1.10.0", err: `no matching versions for query ">=v1.10.0"`},
	{path: queryRepo, query: "6cf84eb", vers: "v0.0.2-0.20180704023347-6cf84ebaea54"},

	// golang.org/issue/27173: A pseudo-version may be based on the highest tag on
	// any parent commit, or any existing semantically-lower tag: a given commit
	// could have been a pre-release for a backport tag at any point.
	{path: queryRepo, query: "3ef0cec634e0", vers: "v0.1.2-0.20180704023347-3ef0cec634e0"},
	{path: queryRepo, query: "v0.1.2-0.20180704023347-3ef0cec634e0", vers: "v0.1.2-0.20180704023347-3ef0cec634e0"},
	{path: queryRepo, query: "v0.1.1-0.20180704023347-3ef0cec634e0", vers: "v0.1.1-0.20180704023347-3ef0cec634e0"},
	{path: queryRepo, query: "v0.0.4-0.20180704023347-3ef0cec634e0", vers: "v0.0.4-0.20180704023347-3ef0cec634e0"},

	// Invalid tags are tested in cmd/go/testdata/script/mod_pseudo_invalid.txt.

	{path: queryRepo, query: "start", vers: "v0.0.0-20180704023101-5e9e31667ddf"},
	{path: queryRepo, query: "5e9e31667ddf", vers: "v0.0.0-20180704023101-5e9e31667ddf"},
	{path: queryRepo, query: "v0.0.0-20180704023101-5e9e31667ddf", vers: "v0.0.0-20180704023101-5e9e31667ddf"},

	{path: queryRepo, query: "7a1b6bf", vers: "v0.1.0"},

	{path: queryRepoV2, query: "<v0.0.0", err: `no matching versions for query "<v0.0.0"`},
	{path: queryRepoV2, query: "<=v0.0.0", err: `no matching versions for query "<=v0.0.0"`},
	{path: queryRepoV2, query: ">v0.0.0", vers: "v2.0.0"},
	{path: queryRepoV2, query: ">=v0.0.0", vers: "v2.0.0"},

	{path: queryRepoV2, query: "v2", vers: "v2.5.5"},
	{path: queryRepoV2, query: "v2.5", vers: "v2.5.5"},
	{path: queryRepoV2, query: "v2.6", err: `no matching versions for query "v2.6"`},
	{path: queryRepoV2, query: "v2.6.0-pre1", vers: "v2.6.0-pre1"},
	{path: queryRepoV2, query: "latest", vers: "v2.5.5"},

	// Commit e0cf3de987e6 is actually v1.19.10-pre1, not anything resembling v3,
	// and it has a go.mod file with a non-v3 module path. Attempting to query it
	// as the v3 module should fail.
	{path: queryRepoV3, query: "e0cf3de987e6", err: `vcs-test.golang.org/git/querytest.git/v3@v3.0.0-20180704024501-e0cf3de987e6: invalid version: go.mod has non-.../v3 module path "vcs-test.golang.org/git/querytest.git" (and .../v3/go.mod does not exist) at revision e0cf3de987e6`},

	// The querytest repo does not have any commits tagged with major version 3,
	// and the latest commit in the repo has a go.mod file specifying a non-v3 path.
	// That should prevent us from resolving any version for the /v3 path.
	{path: queryRepoV3, query: "latest", err: `no matching versions for query "latest"`},

	{path: emptyRepoPath, query: "latest", vers: "v0.0.0-20180704023549-7bb914627242"},
	{path: emptyRepoPath, query: ">v0.0.0", err: `no matching versions for query ">v0.0.0"`},
	{path: emptyRepoPath, query: "<v10.0.0", err: `no matching versions for query "<v10.0.0"`},
}

func TestQuery(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)
	testenv.MustHaveExecPath(t, "git")

	ctx := context.Background()

	for _, tt := range queryTests {
		loaderstate := NewState()
		allow := tt.allow
		if allow == "" {
			allow = "*"
		}
		allowed := func(ctx context.Context, m module.Version) error {
			if ok, _ := path.Match(allow, m.Version); !ok {
				return module.VersionError(m, ErrDisallowed)
			}
			return nil
		}
		tt := tt
		t.Run(strings.ReplaceAll(tt.path, "/", "_")+"/"+tt.query+"/"+tt.current+"/"+allow, func(t *testing.T) {
			t.Parallel()

			info, err := Query(loaderstate, ctx, tt.path, tt.query, tt.current, allowed)
			if tt.err != "" {
				if err == nil {
					t.Errorf("Query(_, %q, %q, %q, %v) = %v, want error %q", tt.path, tt.query, tt.current, allow, info.Version, tt.err)
				} else if err.Error() != tt.err {
					t.Errorf("Query(_, %q, %q, %q, %v): %v\nwant error %q", tt.path, tt.query, tt.current, allow, err, tt.err)
				}
				return
			}
			if err != nil {
				t.Fatalf("Query(_, %q, %q, %q, %v): %v\nwant %v", tt.path, tt.query, tt.current, allow, err, tt.vers)
			}
			if info.Version != tt.vers {
				t.Errorf("Query(_, %q, %q, %q, %v) = %v, want %v", tt.path, tt.query, tt.current, allow, info.Version, tt.vers)
			}
		})
	}
}
