// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sumweb

import (
	"bytes"
	"fmt"
	"strings"
	"sync"
	"testing"

	"cmd/go/internal/note"
	"cmd/go/internal/tlog"
)

const (
	testName        = "localhost.localdev/sumdb"
	testVerifierKey = "localhost.localdev/sumdb+00000c67+AcTrnkbUA+TU4heY3hkjiSES/DSQniBqIeQ/YppAUtK6"
	testSignerKey   = "PRIVATE+KEY+localhost.localdev/sumdb+00000c67+AXu6+oaVaOYuQOFrf1V59JK1owcFlJcHwwXHDfDGxSPk"
)

func TestConnLookup(t *testing.T) {
	tc := newTestClient(t)
	tc.mustHaveLatest(1)

	// Basic lookup.
	tc.mustLookup("rsc.io/sampler", "v1.3.0", "rsc.io/sampler v1.3.0 h1:7uVkIFmeBqHfdjD+gZwtXXI+RODJ2Wc4O7MPEh/QiW4=")
	tc.mustHaveLatest(3)

	// Everything should now be cached, both for the original package and its /go.mod.
	tc.getOK = false
	tc.mustLookup("rsc.io/sampler", "v1.3.0", "rsc.io/sampler v1.3.0 h1:7uVkIFmeBqHfdjD+gZwtXXI+RODJ2Wc4O7MPEh/QiW4=")
	tc.mustLookup("rsc.io/sampler", "v1.3.0/go.mod", "rsc.io/sampler v1.3.0/go.mod h1:T1hPZKmBbMNahiBKFy5HrXp6adAjACjK9JXDnKaTXpA=")
	tc.mustHaveLatest(3)
	tc.getOK = true
	tc.getTileOK = false // the cache has what we need

	// Lookup with multiple returned lines.
	tc.mustLookup("rsc.io/quote", "v1.5.2", "rsc.io/quote v1.5.2 h1:w5fcysjrx7yqtD/aO+QwRjYZOKnaM9Uh2b40tElTs3Y=\nrsc.io/quote v1.5.2 h2:xyzzy")
	tc.mustHaveLatest(3)

	// Lookup with need for !-encoding.
	// rsc.io/Quote is the only record written after rsc.io/samper,
	// so it is the only one that should need more tiles.
	tc.getTileOK = true
	tc.mustLookup("rsc.io/Quote", "v1.5.2", "rsc.io/Quote v1.5.2 h1:uppercase!=")
	tc.mustHaveLatest(4)
}

func TestConnBadTiles(t *testing.T) {
	tc := newTestClient(t)

	flipBits := func() {
		for url, data := range tc.remote {
			if strings.Contains(url, "/tile/") {
				for i := range data {
					data[i] ^= 0x80
				}
			}
		}
	}

	// Bad tiles in initial download.
	tc.mustHaveLatest(1)
	flipBits()
	_, err := tc.conn.Lookup("rsc.io/sampler", "v1.3.0")
	tc.mustError(err, "rsc.io/sampler@v1.3.0: initializing sumweb.Conn: checking tree#1: downloaded inconsistent tile")
	flipBits()
	tc.newConn()
	tc.mustLookup("rsc.io/sampler", "v1.3.0", "rsc.io/sampler v1.3.0 h1:7uVkIFmeBqHfdjD+gZwtXXI+RODJ2Wc4O7MPEh/QiW4=")

	// Bad tiles after initial download.
	flipBits()
	_, err = tc.conn.Lookup("rsc.io/Quote", "v1.5.2")
	tc.mustError(err, "rsc.io/Quote@v1.5.2: checking tree#3 against tree#4: downloaded inconsistent tile")
	flipBits()
	tc.newConn()
	tc.mustLookup("rsc.io/Quote", "v1.5.2", "rsc.io/Quote v1.5.2 h1:uppercase!=")

	// Bad starting tree hash looks like bad tiles.
	tc.newConn()
	text := tlog.FormatTree(tlog.Tree{N: 1, Hash: tlog.Hash{}})
	data, err := note.Sign(&note.Note{Text: string(text)}, tc.signer)
	if err != nil {
		tc.t.Fatal(err)
	}
	tc.config[testName+"/latest"] = data
	_, err = tc.conn.Lookup("rsc.io/sampler", "v1.3.0")
	tc.mustError(err, "rsc.io/sampler@v1.3.0: initializing sumweb.Conn: checking tree#1: downloaded inconsistent tile")
}

func TestConnFork(t *testing.T) {
	tc := newTestClient(t)
	tc2 := tc.fork()

	tc.addRecord("rsc.io/pkg1@v1.5.2", `rsc.io/pkg1 v1.5.2 h1:hash!=
`)
	tc.addRecord("rsc.io/pkg1@v1.5.4", `rsc.io/pkg1 v1.5.4 h1:hash!=
`)
	tc.mustLookup("rsc.io/pkg1", "v1.5.2", "rsc.io/pkg1 v1.5.2 h1:hash!=")

	tc2.addRecord("rsc.io/pkg1@v1.5.3", `rsc.io/pkg1 v1.5.3 h1:hash!=
`)
	tc2.addRecord("rsc.io/pkg1@v1.5.4", `rsc.io/pkg1 v1.5.4 h1:hash!=
`)
	tc2.mustLookup("rsc.io/pkg1", "v1.5.4", "rsc.io/pkg1 v1.5.4 h1:hash!=")

	key := "/lookup/rsc.io/pkg1@v1.5.2"
	tc2.remote[key] = tc.remote[key]
	_, err := tc2.conn.Lookup("rsc.io/pkg1", "v1.5.2")
	tc2.mustError(err, ErrSecurity.Error())

	/*
	   SECURITY ERROR
	   go.sum database server misbehavior detected!

	   old database:
	   	go.sum database tree!
	   	5
	   	nWzN20+pwMt62p7jbv1/NlN95ePTlHijabv5zO/s36w=

	   	— localhost.localdev/sumdb AAAMZ5/2FVAdMH58kmnz/0h299pwyskEbzDzoa2/YaPdhvLya4YWDFQQxu2TQb5GpwAH4NdWnTwuhILafisyf3CNbgg=

	   new database:
	   	go.sum database tree
	   	6
	   	wc4SkQt52o5W2nQ8To2ARs+mWuUJjss+sdleoiqxMmM=

	   	— localhost.localdev/sumdb AAAMZ6oRNswlEZ6ZZhxrCvgl1MBy+nusq4JU+TG6Fe2NihWLqOzb+y2c2kzRLoCr4tvw9o36ucQEnhc20e4nA4Qc/wc=

	   proof of misbehavior:
	   	T7i+H/8ER4nXOiw4Bj0koZOkGjkxoNvlI34GpvhHhQg=
	   	Nsuejv72de9hYNM5bqFv8rv3gm3zJQwv/DT/WNbLDLA=
	   	mOmqqZ1aI/lzS94oq/JSbj7pD8Rv9S+xDyi12BtVSHo=
	   	/7Aw5jVSMM9sFjQhaMg+iiDYPMk6decH7QLOGrL9Lx0=
	*/

	wants := []string{
		"SECURITY ERROR",
		"go.sum database server misbehavior detected!",
		"old database:\n\tgo.sum database tree\n\t5\n",
		"— localhost.localdev/sumdb AAAMZ5/2FVAd",
		"new database:\n\tgo.sum database tree\n\t6\n",
		"— localhost.localdev/sumdb AAAMZ6oRNswl",
		"proof of misbehavior:\n\tT7i+H/8ER4nXOiw4Bj0k",
	}
	text := tc2.security.String()
	for _, want := range wants {
		if !strings.Contains(text, want) {
			t.Fatalf("cannot find %q in security text:\n%s", want, text)
		}
	}
}

func TestConnGONOSUMDB(t *testing.T) {
	tc := newTestClient(t)
	tc.conn.SetGONOSUMDB("p,*/q")
	tc.conn.Lookup("rsc.io/sampler", "v1.3.0") // initialize before we turn off network
	tc.getOK = false

	ok := []string{
		"abc",
		"a/p",
		"pq",
		"q",
		"n/o/p/q",
	}
	skip := []string{
		"p",
		"p/x",
		"x/q",
		"x/q/z",
	}

	for _, path := range ok {
		_, err := tc.conn.Lookup(path, "v1.0.0")
		if err == ErrGONOSUMDB {
			t.Errorf("Lookup(%q): ErrGONOSUMDB, wanted failed actual lookup", path)
		}
	}
	for _, path := range skip {
		_, err := tc.conn.Lookup(path, "v1.0.0")
		if err != ErrGONOSUMDB {
			t.Errorf("Lookup(%q): %v, wanted ErrGONOSUMDB", path, err)
		}
	}
}

// A testClient is a self-contained client-side testing environment.
type testClient struct {
	t          *testing.T // active test
	conn       *Conn      // conn being tested
	tileHeight int        // tile height to use (default 2)
	getOK      bool       // should tc.GetURL succeed?
	getTileOK  bool       // should tc.GetURL of tiles succeed?
	treeSize   int64
	hashes     []tlog.Hash
	remote     map[string][]byte
	signer     note.Signer

	// mu protects config, cache, log, security
	// during concurrent use of the exported methods
	// by the conn itself (testClient is the Conn's Client,
	// and the Client methods can both read and write these fields).
	// Unexported methods invoked directly by the test
	// (for example, addRecord) need not hold the mutex:
	// for proper test execution those methods should only
	// be called when the Conn is idle and not using its Client.
	// Not holding the mutex in those methods ensures
	// that if a mistake is made, go test -race will report it.
	// (Holding the mutex would eliminate the race report but
	// not the underlying problem.)
	// Similarly, the get map is not protected by the mutex,
	// because the Client methods only read it.
	mu       sync.Mutex // prot
	config   map[string][]byte
	cache    map[string][]byte
	security bytes.Buffer
}

// newTestClient returns a new testClient that will call t.Fatal on error
// and has a few records already available on the remote server.
func newTestClient(t *testing.T) *testClient {
	tc := &testClient{
		t:          t,
		tileHeight: 2,
		getOK:      true,
		getTileOK:  true,
		config:     make(map[string][]byte),
		cache:      make(map[string][]byte),
		remote:     make(map[string][]byte),
	}

	tc.config["key"] = []byte(testVerifierKey + "\n")
	var err error
	tc.signer, err = note.NewSigner(testSignerKey)
	if err != nil {
		t.Fatal(err)
	}

	tc.newConn()

	tc.addRecord("rsc.io/quote@v1.5.2", `rsc.io/quote v1.5.2 h1:w5fcysjrx7yqtD/aO+QwRjYZOKnaM9Uh2b40tElTs3Y=
rsc.io/quote v1.5.2/go.mod h1:LzX7hefJvL54yjefDEDHNONDjII0t9xZLPXsUe+TKr0=
rsc.io/quote v1.5.2 h2:xyzzy
`)

	tc.addRecord("golang.org/x/text@v0.0.0-20170915032832-14c0d48ead0c", `golang.org/x/text v0.0.0-20170915032832-14c0d48ead0c h1:qgOY6WgZOaTkIIMiVjBQcw93ERBE4m30iBm00nkL0i8=
golang.org/x/text v0.0.0-20170915032832-14c0d48ead0c/go.mod h1:NqM8EUOU14njkJ3fqMW+pc6Ldnwhi/IjpwHt7yyuwOQ=
`)
	tc.addRecord("rsc.io/sampler@v1.3.0", `rsc.io/sampler v1.3.0 h1:7uVkIFmeBqHfdjD+gZwtXXI+RODJ2Wc4O7MPEh/QiW4=
rsc.io/sampler v1.3.0/go.mod h1:T1hPZKmBbMNahiBKFy5HrXp6adAjACjK9JXDnKaTXpA=
`)
	tc.config[testName+"/latest"] = tc.signTree(1)

	tc.addRecord("rsc.io/!quote@v1.5.2", `rsc.io/Quote v1.5.2 h1:uppercase!=
`)
	return tc
}

// newConn resets the Conn associated with tc.
// This clears any in-memory cache from the Conn
// but not tc's on-disk cache.
func (tc *testClient) newConn() {
	tc.conn = NewConn(tc)
	tc.conn.SetTileHeight(tc.tileHeight)
}

// mustLookup does a lookup for path@vers and checks that the lines that come back match want.
func (tc *testClient) mustLookup(path, vers, want string) {
	tc.t.Helper()
	lines, err := tc.conn.Lookup(path, vers)
	if err != nil {
		tc.t.Fatal(err)
	}
	if strings.Join(lines, "\n") != want {
		tc.t.Fatalf("Lookup(%q, %q):\n\t%s\nwant:\n\t%s", path, vers, strings.Join(lines, "\n\t"), strings.Replace(want, "\n", "\n\t", -1))
	}
}

// mustHaveLatest checks that the on-disk configuration
// for latest is a tree of size n.
func (tc *testClient) mustHaveLatest(n int64) {
	tc.t.Helper()

	latest := tc.config[testName+"/latest"]
	lines := strings.Split(string(latest), "\n")
	if len(lines) < 2 || lines[1] != fmt.Sprint(n) {
		tc.t.Fatalf("/latest should have tree %d, but has:\n%s", n, latest)
	}
}

// mustError checks that err's error string contains the text.
func (tc *testClient) mustError(err error, text string) {
	tc.t.Helper()
	if err == nil || !strings.Contains(err.Error(), text) {
		tc.t.Fatalf("err = %v, want %q", err, text)
	}
}

// fork returns a copy of tc.
// Changes made to the new copy or to tc are not reflected in the other.
func (tc *testClient) fork() *testClient {
	tc2 := &testClient{
		t:          tc.t,
		getOK:      tc.getOK,
		getTileOK:  tc.getTileOK,
		tileHeight: tc.tileHeight,
		treeSize:   tc.treeSize,
		hashes:     append([]tlog.Hash{}, tc.hashes...),
		signer:     tc.signer,
		config:     copyMap(tc.config),
		cache:      copyMap(tc.cache),
		remote:     copyMap(tc.remote),
	}
	tc2.newConn()
	return tc2
}

func copyMap(m map[string][]byte) map[string][]byte {
	m2 := make(map[string][]byte)
	for k, v := range m {
		m2[k] = v
	}
	return m2
}

// ReadHashes is tc's implementation of tlog.HashReader, for use with
// tlog.TreeHash and so on.
func (tc *testClient) ReadHashes(indexes []int64) ([]tlog.Hash, error) {
	var list []tlog.Hash
	for _, id := range indexes {
		list = append(list, tc.hashes[id])
	}
	return list, nil
}

// addRecord adds a log record using the given (!-encoded) key and data.
func (tc *testClient) addRecord(key, data string) {
	tc.t.Helper()

	// Create record, add hashes to log tree.
	id := tc.treeSize
	tc.treeSize++
	rec, err := tlog.FormatRecord(id, []byte(data))
	if err != nil {
		tc.t.Fatal(err)
	}
	hashes, err := tlog.StoredHashesForRecordHash(id, tlog.RecordHash([]byte(data)), tc)
	if err != nil {
		tc.t.Fatal(err)
	}
	tc.hashes = append(tc.hashes, hashes...)

	// Create lookup result.
	tc.remote["/lookup/"+key] = append(rec, tc.signTree(tc.treeSize)...)

	// Create new tiles.
	tiles := tlog.NewTiles(tc.tileHeight, id, tc.treeSize)
	for _, tile := range tiles {
		data, err := tlog.ReadTileData(tile, tc)
		if err != nil {
			tc.t.Fatal(err)
		}
		tc.remote["/"+tile.Path()] = data
		// TODO delete old partial tiles
	}
}

// signTree returns the signed head for the tree of the given size.
func (tc *testClient) signTree(size int64) []byte {
	h, err := tlog.TreeHash(size, tc)
	if err != nil {
		tc.t.Fatal(err)
	}
	text := tlog.FormatTree(tlog.Tree{N: size, Hash: h})
	data, err := note.Sign(&note.Note{Text: string(text)}, tc.signer)
	if err != nil {
		tc.t.Fatal(err)
	}
	return data
}

// ReadRemote is for tc's implementation of Client.
func (tc *testClient) ReadRemote(path string) ([]byte, error) {
	// No mutex here because only the Client should be running
	// and the Client cannot change tc.get.
	if !tc.getOK {
		return nil, fmt.Errorf("disallowed remote read %s", path)
	}
	if strings.Contains(path, "/tile/") && !tc.getTileOK {
		return nil, fmt.Errorf("disallowed remote tile read %s", path)
	}

	data, ok := tc.remote[path]
	if !ok {
		return nil, fmt.Errorf("no remote path %s", path)
	}
	return data, nil
}

// ReadConfig is for tc's implementation of Client.
func (tc *testClient) ReadConfig(file string) ([]byte, error) {
	tc.mu.Lock()
	defer tc.mu.Unlock()

	data, ok := tc.config[file]
	if !ok {
		return nil, fmt.Errorf("no config %s", file)
	}
	return data, nil
}

// WriteConfig is for tc's implementation of Client.
func (tc *testClient) WriteConfig(file string, old, new []byte) error {
	tc.mu.Lock()
	defer tc.mu.Unlock()

	data := tc.config[file]
	if !bytes.Equal(old, data) {
		return ErrWriteConflict
	}
	tc.config[file] = new
	return nil
}

// ReadCache is for tc's implementation of Client.
func (tc *testClient) ReadCache(file string) ([]byte, error) {
	tc.mu.Lock()
	defer tc.mu.Unlock()

	data, ok := tc.cache[file]
	if !ok {
		return nil, fmt.Errorf("no cache %s", file)
	}
	return data, nil
}

// WriteCache is for tc's implementation of Client.
func (tc *testClient) WriteCache(file string, data []byte) {
	tc.mu.Lock()
	defer tc.mu.Unlock()

	tc.cache[file] = data
}

// Log is for tc's implementation of Client.
func (tc *testClient) Log(msg string) {
	tc.t.Log(msg)
}

// SecurityError is for tc's implementation of Client.
func (tc *testClient) SecurityError(msg string) {
	tc.mu.Lock()
	defer tc.mu.Unlock()

	fmt.Fprintf(&tc.security, "%s\n", strings.TrimRight(msg, "\n"))
}
