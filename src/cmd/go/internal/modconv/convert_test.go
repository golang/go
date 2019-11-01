// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modconv

import (
	"bytes"
	"fmt"
	"internal/testenv"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"cmd/go/internal/cfg"
	"cmd/go/internal/modfetch"
	"cmd/go/internal/modfetch/codehost"

	"golang.org/x/mod/modfile"
	"golang.org/x/mod/module"
)

func TestMain(m *testing.M) {
	os.Exit(testMain(m))
}

func testMain(m *testing.M) int {
	cfg.GOPROXY = "direct"

	if _, err := exec.LookPath("git"); err != nil {
		fmt.Fprintln(os.Stderr, "skipping because git binary not found")
		fmt.Println("PASS")
		return 0
	}

	dir, err := ioutil.TempDir("", "modconv-test-")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(dir)
	modfetch.PkgMod = filepath.Join(dir, "pkg/mod")
	codehost.WorkRoot = filepath.Join(dir, "codework")

	return m.Run()
}

func TestConvertLegacyConfig(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	if testing.Verbose() {
		old := cfg.BuildX
		defer func() {
			cfg.BuildX = old
		}()
		cfg.BuildX = true
	}

	var tests = []struct {
		path  string
		vers  string
		gomod string
	}{
		/*
			Different versions of git seem to find or not find
			github.com/Masterminds/semver's a93e51b5a57e,
			which is an unmerged pull request.
			We'd rather not provide access to unmerged pull requests,
			so the line is removed from the golden file here,
			but some git commands still find it somehow.

			{
				// Gopkg.lock parsing.
				"github.com/golang/dep", "v0.4.0",
				`module github.com/golang/dep

				require (
					github.com/Masterminds/vcs v1.11.1
					github.com/armon/go-radix v0.0.0-20160115234725-4239b77079c7
					github.com/boltdb/bolt v1.3.1
					github.com/go-yaml/yaml v0.0.0-20170407172122-cd8b52f8269e
					github.com/golang/protobuf v0.0.0-20170901042739-5afd06f9d81a
					github.com/jmank88/nuts v0.3.0
					github.com/nightlyone/lockfile v0.0.0-20170707060451-e83dc5e7bba0
					github.com/pelletier/go-toml v0.0.0-20171218135716-b8b5e7696574
					github.com/pkg/errors v0.8.0
					github.com/sdboyer/constext v0.0.0-20170321163424-836a14457353
					golang.org/x/net v0.0.0-20170828231752-66aacef3dd8a
					golang.org/x/sync v0.0.0-20170517211232-f52d1811a629
					golang.org/x/sys v0.0.0-20170830134202-bb24a47a89ea
				)`,
			},
		*/

		// TODO: https://github.com/docker/distribution uses vendor.conf

		{
			// Godeps.json parsing.
			// TODO: Should v2.0.0 work here too?
			"github.com/docker/distribution", "v0.0.0-20150410205453-85de3967aa93",
			`module github.com/docker/distribution

			require (
				github.com/AdRoll/goamz v0.0.0-20150130162828-d3664b76d905
				github.com/MSOpenTech/azure-sdk-for-go v0.0.0-20150323223030-d90753bcad2e
				github.com/Sirupsen/logrus v0.7.3
				github.com/bugsnag/bugsnag-go v1.0.3-0.20141110184014-b1d153021fcd
				github.com/bugsnag/osext v0.0.0-20130617224835-0dd3f918b21b
				github.com/bugsnag/panicwrap v0.0.0-20141110184334-e5f9854865b9
				github.com/codegangsta/cli v1.4.2-0.20150131031259-6086d7927ec3
				github.com/docker/docker v1.4.2-0.20150204013315-165ea5c158cf
				github.com/docker/libtrust v0.0.0-20150114040149-fa567046d9b1
				github.com/garyburd/redigo v0.0.0-20150301180006-535138d7bcd7
				github.com/gorilla/context v0.0.0-20140604161150-14f550f51af5
				github.com/gorilla/handlers v0.0.0-20140825150757-0e84b7d810c1
				github.com/gorilla/mux v0.0.0-20140926153814-e444e69cbd2e
				github.com/jlhawn/go-crypto v0.0.0-20150401213827-cd738dde20f0
				github.com/yvasiyarov/go-metrics v0.0.0-20140926110328-57bccd1ccd43
				github.com/yvasiyarov/gorelic v0.0.7-0.20141212073537-a9bba5b9ab50
				github.com/yvasiyarov/newrelic_platform_go v0.0.0-20140908184405-b21fdbd4370f
				golang.org/x/net v0.0.0-20150202051010-1dfe7915deaf
				gopkg.in/check.v1 v1.0.0-20141024133853-64131543e789
				gopkg.in/yaml.v2 v2.0.0-20150116202057-bef53efd0c76
			)`,
		},

		{
			// golang.org/issue/24585 - confusion about v2.0.0 tag in legacy non-v2 module
			"github.com/fishy/gcsbucket", "v0.0.0-20180217031846-618d60fe84e0",
			`module github.com/fishy/gcsbucket

			require (
				cloud.google.com/go v0.18.0
				github.com/fishy/fsdb v0.0.0-20180217030800-5527ded01371
				github.com/golang/protobuf v1.0.0
				github.com/googleapis/gax-go v2.0.0+incompatible
				golang.org/x/net v0.0.0-20180216171745-136a25c244d3
				golang.org/x/oauth2 v0.0.0-20180207181906-543e37812f10
				golang.org/x/text v0.3.1-0.20180208041248-4e4a3210bb54
				google.golang.org/api v0.0.0-20180217000815-c7a403bb5fe1
				google.golang.org/appengine v1.0.0
				google.golang.org/genproto v0.0.0-20180206005123-2b5a72b8730b
				google.golang.org/grpc v1.10.0
			)`,
		},
	}

	for _, tt := range tests {
		t.Run(strings.ReplaceAll(tt.path, "/", "_")+"_"+tt.vers, func(t *testing.T) {
			f, err := modfile.Parse("golden", []byte(tt.gomod), nil)
			if err != nil {
				t.Fatal(err)
			}
			want, err := f.Format()
			if err != nil {
				t.Fatal(err)
			}

			dir, err := modfetch.Download(module.Version{Path: tt.path, Version: tt.vers})
			if err != nil {
				t.Fatal(err)
			}

			for name := range Converters {
				file := filepath.Join(dir, name)
				data, err := ioutil.ReadFile(file)
				if err == nil {
					f := new(modfile.File)
					f.AddModuleStmt(tt.path)
					if err := ConvertLegacyConfig(f, filepath.ToSlash(file), data); err != nil {
						t.Fatal(err)
					}
					out, err := f.Format()
					if err != nil {
						t.Fatalf("format after conversion: %v", err)
					}
					if !bytes.Equal(out, want) {
						t.Fatalf("final go.mod:\n%s\n\nwant:\n%s", out, want)
					}
					return
				}
			}
			t.Fatalf("no converter found for %s@%s", tt.path, tt.vers)
		})
	}
}
