// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sumweb

import (
	"bytes"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"

	"cmd/go/internal/note"
	"cmd/go/internal/str"
	"cmd/go/internal/tlog"
)

// A Client provides the external operations
// (file caching, HTTP fetches, and so on)
// needed to implement the HTTP client Conn.
// The methods must be safe for concurrent use by multiple goroutines.
type Client interface {
	// ReadRemote reads and returns the content served at the given path
	// on the remote database server. The path begins with "/lookup" or "/tile/".
	// It is the implementation's responsibility to turn that path into a full URL
	// and make the HTTP request. ReadRemote should return an error for
	// any non-200 HTTP response status.
	ReadRemote(path string) ([]byte, error)

	// ReadConfig reads and returns the content of the named configuration file.
	// There are only a fixed set of configuration files.
	//
	// "key" returns a file containing the verifier key for the server.
	//
	// serverName + "/latest" returns a file containing the latest known
	// signed tree from the server. It is read and written (using WriteConfig).
	// To signal that the client wishes to start with an "empty" signed tree,
	// ReadConfig can return a successful empty result (0 bytes of data).
	ReadConfig(file string) ([]byte, error)

	// WriteConfig updates the content of the named configuration file,
	// changing it from the old []byte to the new []byte.
	// If the old []byte does not match the stored configuration,
	// WriteConfig must return ErrWriteConflict.
	// Otherwise, WriteConfig should atomically replace old with new.
	WriteConfig(file string, old, new []byte) error

	// ReadCache reads and returns the content of the named cache file.
	// Any returned error will be treated as equivalent to the file not existing.
	// There can be arbitrarily many cache files, such as:
	//	serverName/lookup/pkg@version
	//	serverName/tile/8/1/x123/456
	ReadCache(file string) ([]byte, error)

	// WriteCache writes the named cache file.
	WriteCache(file string, data []byte)

	// Log prints the given log message (such as with log.Print)
	Log(msg string)

	// SecurityError prints the given security error log message.
	// The Conn returns ErrSecurity from any operation that invokes SecurityError,
	// but the return value is mainly for testing. In a real program,
	// SecurityError should typically print the message and call log.Fatal or os.Exit.
	SecurityError(msg string)
}

// ErrWriteConflict signals a write conflict during Client.WriteConfig.
var ErrWriteConflict = errors.New("write conflict")

// ErrSecurity is returned by Conn operations that invoke Client.SecurityError.
var ErrSecurity = errors.New("security error: misbehaving server")

// A Conn is a client connection to a go.sum database.
// All the methods are safe for simultaneous use by multiple goroutines.
type Conn struct {
	client Client // client-provided external world

	didLookup uint32

	// one-time initialized data
	initOnce   sync.Once
	initErr    error          // init error, if any
	name       string         // name of accepted verifier
	verifiers  note.Verifiers // accepted verifiers (just one, but Verifiers for note.Open)
	tileReader tileReader
	tileHeight int
	nosumdb    string

	record    parCache // cache of record lookup, keyed by path@vers
	tileCache parCache // cache of c.readTile, keyed by tile

	latestMu  sync.Mutex
	latest    tlog.Tree // latest known tree head
	latestMsg []byte    // encoded signed note for latest

	tileSavedMu sync.Mutex
	tileSaved   map[tlog.Tile]bool // which tiles have been saved using c.client.WriteCache already
}

// NewConn returns a new Conn using the given Client.
func NewConn(client Client) *Conn {
	return &Conn{
		client: client,
	}
}

// init initiailzes the conn (if not already initialized)
// and returns any initialization error.
func (c *Conn) init() error {
	c.initOnce.Do(c.initWork)
	return c.initErr
}

// initWork does the actual initialization work.
func (c *Conn) initWork() {
	defer func() {
		if c.initErr != nil {
			c.initErr = fmt.Errorf("initializing sumweb.Conn: %v", c.initErr)
		}
	}()

	c.tileReader.c = c
	if c.tileHeight == 0 {
		c.tileHeight = 8
	}
	c.tileSaved = make(map[tlog.Tile]bool)

	vkey, err := c.client.ReadConfig("key")
	if err != nil {
		c.initErr = err
		return
	}
	verifier, err := note.NewVerifier(strings.TrimSpace(string(vkey)))
	if err != nil {
		c.initErr = err
		return
	}
	c.verifiers = note.VerifierList(verifier)
	c.name = verifier.Name()

	data, err := c.client.ReadConfig(c.name + "/latest")
	if err != nil {
		c.initErr = err
		return
	}
	if err := c.mergeLatest(data); err != nil {
		c.initErr = err
		return
	}
}

// SetTileHeight sets the tile height for the Conn.
// Any call to SetTileHeight must happen before the first call to Lookup.
// If SetTileHeight is not called, the Conn defaults to tile height 8.
func (c *Conn) SetTileHeight(height int) {
	if atomic.LoadUint32(&c.didLookup) != 0 {
		panic("SetTileHeight used after Lookup")
	}
	if c.tileHeight != 0 {
		panic("multiple calls to SetTileHeight")
	}
	c.tileHeight = height
}

// SetGONOSUMDB sets the list of comma-separated GONOSUMDB patterns for the Conn.
// For any module path matching one of the patterns,
// Lookup will return ErrGONOSUMDB.
// Any call to SetGONOSUMDB must happen before the first call to Lookup.
func (c *Conn) SetGONOSUMDB(list string) {
	if atomic.LoadUint32(&c.didLookup) != 0 {
		panic("SetGONOSUMDB used after Lookup")
	}
	if c.nosumdb != "" {
		panic("multiple calls to SetGONOSUMDB")
	}
	c.nosumdb = list
}

// ErrGONOSUMDB is returned by Lookup for paths that match
// a pattern listed in the GONOSUMDB list (set by SetGONOSUMDB,
// usually from the environment variable).
var ErrGONOSUMDB = errors.New("skipped (listed in GONOSUMDB)")

func (c *Conn) skip(target string) bool {
	return str.GlobsMatchPath(c.nosumdb, target)
}

// Lookup returns the go.sum lines for the given module path and version.
// The version may end in a /go.mod suffix, in which case Lookup returns
// the go.sum lines for the module's go.mod-only hash.
func (c *Conn) Lookup(path, vers string) (lines []string, err error) {
	atomic.StoreUint32(&c.didLookup, 1)

	if c.skip(path) {
		return nil, ErrGONOSUMDB
	}

	defer func() {
		if err != nil {
			err = fmt.Errorf("%s@%s: %v", path, vers, err)
		}
	}()

	if err := c.init(); err != nil {
		return nil, err
	}

	// Prepare encoded cache filename / URL.
	epath, err := encodePath(path)
	if err != nil {
		return nil, err
	}
	evers, err := encodeVersion(strings.TrimSuffix(vers, "/go.mod"))
	if err != nil {
		return nil, err
	}
	file := c.name + "/lookup/" + epath + "@" + evers
	remotePath := "/lookup/" + epath + "@" + evers

	// Fetch the data.
	// The lookupCache avoids redundant ReadCache/GetURL operations
	// (especially since go.sum lines tend to come in pairs for a given
	// path and version) and also avoids having multiple of the same
	// request in flight at once.
	type cached struct {
		data []byte
		err  error
	}
	result := c.record.Do(file, func() interface{} {
		// Try the on-disk cache, or else get from web.
		writeCache := false
		data, err := c.client.ReadCache(file)
		if err != nil {
			data, err = c.client.ReadRemote(remotePath)
			if err != nil {
				return cached{nil, err}
			}
			writeCache = true
		}

		// Validate the record before using it for anything.
		id, text, treeMsg, err := tlog.ParseRecord(data)
		if err != nil {
			return cached{nil, err}
		}
		if err := c.mergeLatest(treeMsg); err != nil {
			return cached{nil, err}
		}
		if err := c.checkRecord(id, text); err != nil {
			return cached{nil, err}
		}

		// Now that we've validated the record,
		// save it to the on-disk cache (unless that's where it came from).
		if writeCache {
			c.client.WriteCache(file, data)
		}

		return cached{data, nil}
	}).(cached)
	if result.err != nil {
		return nil, result.err
	}

	// Extract the lines for the specific version we want
	// (with or without /go.mod).
	prefix := path + " " + vers + " "
	var hashes []string
	for _, line := range strings.Split(string(result.data), "\n") {
		if strings.HasPrefix(line, prefix) {
			hashes = append(hashes, line)
		}
	}
	return hashes, nil
}

// mergeLatest merges the tree head in msg
// with the Conn's current latest tree head,
// ensuring the result is a consistent timeline.
// If the result is inconsistent, mergeLatest calls c.client.SecurityError
// with a detailed security error message and then
// (only if c.client.SecurityError does not exit the program) returns ErrSecurity.
// If the Conn's current latest tree head moves forward,
// mergeLatest updates the underlying configuration file as well,
// taking care to merge any independent updates to that configuration.
func (c *Conn) mergeLatest(msg []byte) error {
	// Merge msg into our in-memory copy of the latest tree head.
	when, err := c.mergeLatestMem(msg)
	if err != nil {
		return err
	}
	if when != msgFuture {
		// msg matched our present or was in the past.
		// No change to our present, so no update of config file.
		return nil
	}

	// Flush our extended timeline back out to the configuration file.
	// If the configuration file has been updated in the interim,
	// we need to merge any updates made there as well.
	// Note that writeConfig is an atomic compare-and-swap.
	for {
		msg, err := c.client.ReadConfig(c.name + "/latest")
		if err != nil {
			return err
		}
		when, err := c.mergeLatestMem(msg)
		if err != nil {
			return err
		}
		if when != msgPast {
			// msg matched our present or was from the future,
			// and now our in-memory copy matches.
			return nil
		}

		// msg (== config) is in the past, so we need to update it.
		c.latestMu.Lock()
		latestMsg := c.latestMsg
		c.latestMu.Unlock()
		if err := c.client.WriteConfig(c.name+"/latest", msg, latestMsg); err != ErrWriteConflict {
			// Success or a non-write-conflict error.
			return err
		}
	}
}

const (
	msgPast = 1 + iota
	msgNow
	msgFuture
)

// mergeLatestMem is like mergeLatest but is only concerned with
// updating the in-memory copy of the latest tree head (c.latest)
// not the configuration file.
// The when result explains when msg happened relative to our
// previous idea of c.latest:
// msgPast means msg was from before c.latest,
// msgNow means msg was exactly c.latest, and
// msgFuture means msg was from after c.latest, which has now been updated.
func (c *Conn) mergeLatestMem(msg []byte) (when int, err error) {
	if len(msg) == 0 {
		// Accept empty msg as the unsigned, empty timeline.
		c.latestMu.Lock()
		latest := c.latest
		c.latestMu.Unlock()
		if latest.N == 0 {
			return msgNow, nil
		}
		return msgPast, nil
	}

	note, err := note.Open(msg, c.verifiers)
	if err != nil {
		return 0, fmt.Errorf("reading tree note: %v\nnote:\n%s", err, msg)
	}
	tree, err := tlog.ParseTree([]byte(note.Text))
	if err != nil {
		return 0, fmt.Errorf("reading tree: %v\ntree:\n%s", err, note.Text)
	}

	// Other lookups may be calling mergeLatest with other heads,
	// so c.latest is changing underfoot. We don't want to hold the
	// c.mu lock during tile fetches, so loop trying to update c.latest.
	c.latestMu.Lock()
	latest := c.latest
	latestMsg := c.latestMsg
	c.latestMu.Unlock()

	for {
		// If the tree head looks old, check that it is on our timeline.
		if tree.N <= latest.N {
			if err := c.checkTrees(tree, msg, latest, latestMsg); err != nil {
				return 0, err
			}
			if tree.N < latest.N {
				return msgPast, nil
			}
			return msgNow, nil
		}

		// The tree head looks new. Check that we are on its timeline and try to move our timeline forward.
		if err := c.checkTrees(latest, latestMsg, tree, msg); err != nil {
			return 0, err
		}

		// Install our msg if possible.
		// Otherwise we will go around again.
		c.latestMu.Lock()
		installed := false
		if c.latest == latest {
			installed = true
			c.latest = tree
			c.latestMsg = msg
		} else {
			latest = c.latest
			latestMsg = c.latestMsg
		}
		c.latestMu.Unlock()

		if installed {
			return msgFuture, nil
		}
	}
}

// checkTrees checks that older (from olderNote) is contained in newer (from newerNote).
// If an error occurs, such as malformed data or a network problem, checkTrees returns that error.
// If on the other hand checkTrees finds evidence of misbehavior, it prepares a detailed
// message and calls log.Fatal.
func (c *Conn) checkTrees(older tlog.Tree, olderNote []byte, newer tlog.Tree, newerNote []byte) error {
	thr := tlog.TileHashReader(newer, &c.tileReader)
	h, err := tlog.TreeHash(older.N, thr)
	if err != nil {
		if older.N == newer.N {
			return fmt.Errorf("checking tree#%d: %v", older.N, err)
		}
		return fmt.Errorf("checking tree#%d against tree#%d: %v", older.N, newer.N, err)
	}
	if h == older.Hash {
		return nil
	}

	// Detected a fork in the tree timeline.
	// Start by reporting the inconsistent signed tree notes.
	var buf bytes.Buffer
	fmt.Fprintf(&buf, "SECURITY ERROR\n")
	fmt.Fprintf(&buf, "go.sum database server misbehavior detected!\n\n")
	indent := func(b []byte) []byte {
		return bytes.Replace(b, []byte("\n"), []byte("\n\t"), -1)
	}
	fmt.Fprintf(&buf, "old database:\n\t%s\n", indent(olderNote))
	fmt.Fprintf(&buf, "new database:\n\t%s\n", indent(newerNote))

	// The notes alone are not enough to prove the inconsistency.
	// We also need to show that the newer note's tree hash for older.N
	// does not match older.Hash. The consumer of this report could
	// of course consult the server to try to verify the inconsistency,
	// but we are holding all the bits we need to prove it right now,
	// so we might as well print them and make the report not depend
	// on the continued availability of the misbehaving server.
	// Preparing this data only reuses the tiled hashes needed for
	// tlog.TreeHash(older.N, thr) above, so assuming thr is caching tiles,
	// there are no new access to the server here, and these operations cannot fail.
	fmt.Fprintf(&buf, "proof of misbehavior:\n\t%v", h)
	if p, err := tlog.ProveTree(newer.N, older.N, thr); err != nil {
		fmt.Fprintf(&buf, "\tinternal error: %v\n", err)
	} else if err := tlog.CheckTree(p, newer.N, newer.Hash, older.N, h); err != nil {
		fmt.Fprintf(&buf, "\tinternal error: generated inconsistent proof\n")
	} else {
		for _, h := range p {
			fmt.Fprintf(&buf, "\n\t%v", h)
		}
	}
	c.client.SecurityError(buf.String())
	return ErrSecurity
}

// checkRecord checks that record #id's hash matches data.
func (c *Conn) checkRecord(id int64, data []byte) error {
	c.latestMu.Lock()
	latest := c.latest
	c.latestMu.Unlock()

	if id >= latest.N {
		return fmt.Errorf("cannot validate record %d in tree of size %d", id, latest.N)
	}
	hashes, err := tlog.TileHashReader(latest, &c.tileReader).ReadHashes([]int64{tlog.StoredHashIndex(0, id)})
	if err != nil {
		return err
	}
	if hashes[0] == tlog.RecordHash(data) {
		return nil
	}
	return fmt.Errorf("cannot authenticate record data in server response")
}

// tileReader is a *Conn wrapper that implements tlog.TileReader.
// The separate type avoids exposing the ReadTiles and SaveTiles
// methods on Conn itself.
type tileReader struct {
	c *Conn
}

func (r *tileReader) Height() int {
	return r.c.tileHeight
}

// ReadTiles reads and returns the requested tiles,
// either from the on-disk cache or the server.
func (r *tileReader) ReadTiles(tiles []tlog.Tile) ([][]byte, error) {
	// Read all the tiles in parallel.
	data := make([][]byte, len(tiles))
	errs := make([]error, len(tiles))
	var wg sync.WaitGroup
	for i, tile := range tiles {
		wg.Add(1)
		go func(i int, tile tlog.Tile) {
			defer wg.Done()
			data[i], errs[i] = r.c.readTile(tile)
		}(i, tile)
	}
	wg.Wait()

	for _, err := range errs {
		if err != nil {
			return nil, err
		}
	}

	return data, nil
}

// tileCacheKey returns the cache key for the tile.
func (c *Conn) tileCacheKey(tile tlog.Tile) string {
	return c.name + "/" + tile.Path()
}

// tileRemotePath returns the remote path for the tile.
func (c *Conn) tileRemotePath(tile tlog.Tile) string {
	return "/" + tile.Path()
}

// readTile reads a single tile, either from the on-disk cache or the server.
func (c *Conn) readTile(tile tlog.Tile) ([]byte, error) {
	type cached struct {
		data []byte
		err  error
	}

	result := c.tileCache.Do(tile, func() interface{} {
		// Try the requested tile in on-disk cache.
		data, err := c.client.ReadCache(c.tileCacheKey(tile))
		if err == nil {
			c.markTileSaved(tile)
			return cached{data, nil}
		}

		// Try the full tile in on-disk cache (if requested tile not already full).
		// We only save authenticated tiles to the on-disk cache,
		// so the recreated prefix is equally authenticated.
		full := tile
		full.W = 1 << tile.H
		if tile != full {
			data, err := c.client.ReadCache(c.tileCacheKey(full))
			if err == nil {
				c.markTileSaved(tile) // don't save tile later; we already have full
				return cached{data[:len(data)/full.W*tile.W], nil}
			}
		}

		// Try requested tile from server.
		data, err = c.client.ReadRemote(c.tileRemotePath(tile))
		if err == nil {
			return cached{data, nil}
		}

		// Try full tile on server.
		// If the partial tile does not exist, it should be because
		// the tile has been completed and only the complete one
		// is available.
		if tile != full {
			data, err := c.client.ReadRemote(c.tileRemotePath(full))
			if err == nil {
				// Note: We could save the full tile in the on-disk cache here,
				// but we don't know if it is valid yet, and we will only find out
				// about the partial data, not the full data. So let SaveTiles
				// save the partial tile, and we'll just refetch the full tile later
				// once we can validate more (or all) of it.
				return cached{data[:len(data)/full.W*tile.W], nil}
			}
		}

		// Nothing worked.
		// Return the error from the server fetch for the requested (not full) tile.
		return cached{nil, err}
	}).(cached)

	return result.data, result.err
}

// markTileSaved records that tile is already present in the on-disk cache,
// so that a future SaveTiles for that tile can be ignored.
func (c *Conn) markTileSaved(tile tlog.Tile) {
	c.tileSavedMu.Lock()
	c.tileSaved[tile] = true
	c.tileSavedMu.Unlock()
}

// SaveTiles saves the now validated tiles.
func (r *tileReader) SaveTiles(tiles []tlog.Tile, data [][]byte) {
	c := r.c

	// Determine which tiles need saving.
	// (Tiles that came from the cache need not be saved back.)
	save := make([]bool, len(tiles))
	c.tileSavedMu.Lock()
	for i, tile := range tiles {
		if !c.tileSaved[tile] {
			save[i] = true
			c.tileSaved[tile] = true
		}
	}
	c.tileSavedMu.Unlock()

	for i, tile := range tiles {
		if save[i] {
			// If WriteCache fails here (out of disk space? i/o error?),
			// c.tileSaved[tile] is still true and we will not try to write it again.
			// Next time we run maybe we'll redownload it again and be
			// more successful.
			c.client.WriteCache(c.name+"/"+tile.Path(), data[i])
		}
	}
}
