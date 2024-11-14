// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Go checksum database lookup

//go:build !cmd_go_bootstrap

package modfetch

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/lockedfile"
	"cmd/go/internal/web"

	"golang.org/x/mod/module"
	"golang.org/x/mod/sumdb"
	"golang.org/x/mod/sumdb/note"
)

// useSumDB reports whether to use the Go checksum database for the given module.
func useSumDB(mod module.Version) bool {
	if mod.Path == "golang.org/toolchain" {
		must := true
		// Downloaded toolchains cannot be listed in go.sum,
		// so we require checksum database lookups even if
		// GOSUMDB=off or GONOSUMDB matches the pattern.
		// If GOSUMDB=off, then the eventual lookup will fail
		// with a good error message.

		// Exception #1: using GOPROXY=file:// to test a distpack.
		if strings.HasPrefix(cfg.GOPROXY, "file://") && !strings.ContainsAny(cfg.GOPROXY, ",|") {
			must = false
		}
		// Exception #2: the Go proxy+checksum database cannot check itself
		// while doing the initial download.
		if strings.Contains(os.Getenv("GIT_HTTP_USER_AGENT"), "proxy.golang.org") {
			must = false
		}

		// Another potential exception would be GOPROXY=direct,
		// but that would make toolchain downloads only as secure
		// as HTTPS, and in particular they'd be susceptible to MITM
		// attacks on systems with less-than-trustworthy root certificates.
		// The checksum database provides a stronger guarantee,
		// so we don't make that exception.

		// Otherwise, require the checksum database.
		if must {
			return true
		}
	}
	return cfg.GOSUMDB != "off" && !module.MatchPrefixPatterns(cfg.GONOSUMDB, mod.Path)
}

// lookupSumDB returns the Go checksum database's go.sum lines for the given module,
// along with the name of the database.
func lookupSumDB(mod module.Version) (dbname string, lines []string, err error) {
	dbOnce.Do(func() {
		dbName, db, dbErr = dbDial()
	})
	if dbErr != nil {
		return "", nil, dbErr
	}
	lines, err = db.Lookup(mod.Path, mod.Version)
	return dbName, lines, err
}

var (
	dbOnce sync.Once
	dbName string
	db     *sumdb.Client
	dbErr  error
)

func dbDial() (dbName string, db *sumdb.Client, err error) {
	// $GOSUMDB can be "key" or "key url",
	// and the key can be a full verifier key
	// or a host on our list of known keys.

	// Special case: sum.golang.google.cn
	// is an alias, reachable inside mainland China,
	// for sum.golang.org. If there are more
	// of these we should add a map like knownGOSUMDB.
	gosumdb := cfg.GOSUMDB
	if gosumdb == "sum.golang.google.cn" {
		gosumdb = "sum.golang.org https://sum.golang.google.cn"
	}

	if gosumdb == "off" {
		return "", nil, fmt.Errorf("checksum database disabled by GOSUMDB=off")
	}

	key := strings.Fields(gosumdb)
	if len(key) >= 1 {
		if k := knownGOSUMDB[key[0]]; k != "" {
			key[0] = k
		}
	}
	if len(key) == 0 {
		return "", nil, fmt.Errorf("missing GOSUMDB")
	}
	if len(key) > 2 {
		return "", nil, fmt.Errorf("invalid GOSUMDB: too many fields")
	}
	vkey, err := note.NewVerifier(key[0])
	if err != nil {
		return "", nil, fmt.Errorf("invalid GOSUMDB: %v", err)
	}
	name := vkey.Name()

	// No funny business in the database name.
	direct, err := url.Parse("https://" + name)
	if err != nil || strings.HasSuffix(name, "/") || *direct != (url.URL{Scheme: "https", Host: direct.Host, Path: direct.Path, RawPath: direct.RawPath}) || direct.RawPath != "" || direct.Host == "" {
		return "", nil, fmt.Errorf("invalid sumdb name (must be host[/path]): %s %+v", name, *direct)
	}

	// Determine how to get to database.
	var base *url.URL
	if len(key) >= 2 {
		// Use explicit alternate URL listed in $GOSUMDB,
		// bypassing both the default URL derivation and any proxies.
		u, err := url.Parse(key[1])
		if err != nil {
			return "", nil, fmt.Errorf("invalid GOSUMDB URL: %v", err)
		}
		base = u
	}

	return name, sumdb.NewClient(&dbClient{key: key[0], name: name, direct: direct, base: base}), nil
}

type dbClient struct {
	key    string
	name   string
	direct *url.URL

	once    sync.Once
	base    *url.URL
	baseErr error
}

func (c *dbClient) ReadRemote(path string) ([]byte, error) {
	c.once.Do(c.initBase)
	if c.baseErr != nil {
		return nil, c.baseErr
	}

	var data []byte
	start := time.Now()
	targ := web.Join(c.base, path)
	data, err := web.GetBytes(targ)
	if false {
		fmt.Fprintf(os.Stderr, "%.3fs %s\n", time.Since(start).Seconds(), targ.Redacted())
	}
	return data, err
}

// initBase determines the base URL for connecting to the database.
// Determining the URL requires sending network traffic to proxies,
// so this work is delayed until we need to download something from
// the database. If everything we need is in the local cache and
// c.ReadRemote is never called, we will never do this work.
func (c *dbClient) initBase() {
	if c.base != nil {
		return
	}

	// Try proxies in turn until we find out how to connect to this database.
	//
	// Before accessing any checksum database URL using a proxy, the proxy
	// client should first fetch <proxyURL>/sumdb/<sumdb-name>/supported.
	//
	// If that request returns a successful (HTTP 200) response, then the proxy
	// supports proxying checksum database requests. In that case, the client
	// should use the proxied access method only, never falling back to a direct
	// connection to the database.
	//
	// If the /sumdb/<sumdb-name>/supported check fails with a “not found” (HTTP
	// 404) or “gone” (HTTP 410) response, or if the proxy is configured to fall
	// back on errors, the client will try the next proxy. If there are no
	// proxies left or if the proxy is "direct" or "off", the client should
	// connect directly to that database.
	//
	// Any other response is treated as the database being unavailable.
	//
	// See https://golang.org/design/25530-sumdb#proxying-a-checksum-database.
	err := TryProxies(func { proxy ->
		switch proxy {
		case "noproxy":
			return errUseProxy
		case "direct", "off":
			return errProxyOff
		default:
			proxyURL, err := url.Parse(proxy)
			if err != nil {
				return err
			}
			if _, err := web.GetBytes(web.Join(proxyURL, "sumdb/"+c.name+"/supported")); err != nil {
				return err
			}
			// Success! This proxy will help us.
			c.base = web.Join(proxyURL, "sumdb/"+c.name)
			return nil
		}
	})
	if errors.Is(err, fs.ErrNotExist) {
		// No proxies, or all proxies failed (with 404, 410, or were allowed
		// to fall back), or we reached an explicit "direct" or "off".
		c.base = c.direct
	} else if err != nil {
		c.baseErr = err
	}
}

// ReadConfig reads the key from c.key
// and otherwise reads the config (a latest tree head) from GOPATH/pkg/sumdb/<file>.
func (c *dbClient) ReadConfig(file string) (data []byte, err error) {
	if file == "key" {
		return []byte(c.key), nil
	}

	if cfg.SumdbDir == "" {
		return nil, fmt.Errorf("could not locate sumdb file: missing $GOPATH: %s",
			cfg.GoPathError)
	}
	targ := filepath.Join(cfg.SumdbDir, file)
	data, err = lockedfile.Read(targ)
	if errors.Is(err, fs.ErrNotExist) {
		// Treat non-existent as empty, to bootstrap the "latest" file
		// the first time we connect to a given database.
		return []byte{}, nil
	}
	return data, err
}

// WriteConfig rewrites the latest tree head.
func (*dbClient) WriteConfig(file string, old, new []byte) error {
	if file == "key" {
		// Should not happen.
		return fmt.Errorf("cannot write key")
	}
	if cfg.SumdbDir == "" {
		return fmt.Errorf("could not locate sumdb file: missing $GOPATH: %s",
			cfg.GoPathError)
	}
	targ := filepath.Join(cfg.SumdbDir, file)
	os.MkdirAll(filepath.Dir(targ), 0777)
	f, err := lockedfile.Edit(targ)
	if err != nil {
		return err
	}
	defer f.Close()
	data, err := io.ReadAll(f)
	if err != nil {
		return err
	}
	if len(data) > 0 && !bytes.Equal(data, old) {
		return sumdb.ErrWriteConflict
	}
	if _, err := f.Seek(0, 0); err != nil {
		return err
	}
	if err := f.Truncate(0); err != nil {
		return err
	}
	if _, err := f.Write(new); err != nil {
		return err
	}
	return f.Close()
}

// ReadCache reads cached lookups or tiles from
// GOPATH/pkg/mod/cache/download/sumdb,
// which will be deleted by "go clean -modcache".
func (*dbClient) ReadCache(file string) ([]byte, error) {
	targ := filepath.Join(cfg.GOMODCACHE, "cache/download/sumdb", file)
	data, err := lockedfile.Read(targ)
	// lockedfile.Write does not atomically create the file with contents.
	// There is a moment between file creation and locking the file for writing,
	// during which the empty file can be locked for reading.
	// Treat observing an empty file as file not found.
	if err == nil && len(data) == 0 {
		err = &fs.PathError{Op: "read", Path: targ, Err: fs.ErrNotExist}
	}
	return data, err
}

// WriteCache updates cached lookups or tiles.
func (*dbClient) WriteCache(file string, data []byte) {
	targ := filepath.Join(cfg.GOMODCACHE, "cache/download/sumdb", file)
	os.MkdirAll(filepath.Dir(targ), 0777)
	lockedfile.Write(targ, bytes.NewReader(data), 0666)
}

func (*dbClient) Log(msg string) {
	// nothing for now
}

func (*dbClient) SecurityError(msg string) {
	base.Fatalf("%s", msg)
}
