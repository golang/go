// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tlog

import (
	"fmt"
	"strconv"
	"strings"
)

// A Tile is a description of a transparency log tile.
// A tile of height H at level L offset N lists W consecutive hashes
// at level H*L of the tree starting at offset N*(2**H).
// A complete tile lists 2**H hashes; a partial tile lists fewer.
// Note that a tile represents the entire subtree of height H
// with those hashes as the leaves. The levels above H*L
// can be reconstructed by hashing the leaves.
//
// Each Tile can be encoded as a “tile coordinate path”
// of the form tile/H/L/NNN[.p/W].
// The .p/W suffix is present only for partial tiles, meaning W < 2**H.
// The NNN element is an encoding of N into 3-digit path elements.
// All but the last path element begins with an "x".
// For example,
// Tile{H: 3, L: 4, N: 1234067, W: 1}'s path
// is tile/3/4/x001/x234/067.p/1, and
// Tile{H: 3, L: 4, N: 1234067, W: 8}'s path
// is tile/3/4/x001/x234/067.
// See the [Tile.Path] method and the [ParseTilePath] function.
//
// The special level L=-1 holds raw record data instead of hashes.
// In this case, the level encodes into a tile path as the path element
// "data" instead of "-1".
//
// See also https://golang.org/design/25530-sumdb#checksum-database
// and https://research.swtch.com/tlog#tiling_a_log.
type Tile struct {
	H int   // height of tile (1 ≤ H ≤ 30)
	L int   // level in tiling (-1 ≤ L ≤ 63)
	N int64 // number within level (0 ≤ N, unbounded)
	W int   // width of tile (1 ≤ W ≤ 2**H; 2**H is complete tile)
}

// TileForIndex returns the tile of fixed height h ≥ 1
// and least width storing the given hash storage index.
//
// If h ≤ 0, [TileForIndex] panics.
func TileForIndex(h int, index int64) Tile {
	if h <= 0 {
		panic(fmt.Sprintf("TileForIndex: invalid height %d", h))
	}
	t, _, _ := tileForIndex(h, index)
	return t
}

// tileForIndex returns the tile of height h ≥ 1
// storing the given hash index, which can be
// reconstructed using tileHash(data[start:end]).
func tileForIndex(h int, index int64) (t Tile, start, end int) {
	level, n := SplitStoredHashIndex(index)
	t.H = h
	t.L = level / h
	level -= t.L * h // now level within tile
	t.N = n << uint(level) >> uint(t.H)
	n -= t.N << uint(t.H) >> uint(level) // now n within tile at level
	t.W = int((n + 1) << uint(level))
	return t, int(n<<uint(level)) * HashSize, int((n+1)<<uint(level)) * HashSize
}

// HashFromTile returns the hash at the given storage index,
// provided that t == TileForIndex(t.H, index) or a wider version,
// and data is t's tile data (of length at least t.W*HashSize).
func HashFromTile(t Tile, data []byte, index int64) (Hash, error) {
	if t.H < 1 || t.H > 30 || t.L < 0 || t.L >= 64 || t.W < 1 || t.W > 1<<uint(t.H) {
		return Hash{}, fmt.Errorf("invalid tile %v", t.Path())
	}
	if len(data) < t.W*HashSize {
		return Hash{}, fmt.Errorf("data len %d too short for tile %v", len(data), t.Path())
	}
	t1, start, end := tileForIndex(t.H, index)
	if t.L != t1.L || t.N != t1.N || t.W < t1.W {
		return Hash{}, fmt.Errorf("index %v is in %v not %v", index, t1.Path(), t.Path())
	}
	return tileHash(data[start:end]), nil
}

// tileHash computes the subtree hash corresponding to the (2^K)-1 hashes in data.
func tileHash(data []byte) Hash {
	if len(data) == 0 {
		panic("bad math in tileHash")
	}
	if len(data) == HashSize {
		var h Hash
		copy(h[:], data)
		return h
	}
	n := len(data) / 2
	return NodeHash(tileHash(data[:n]), tileHash(data[n:]))
}

// NewTiles returns the coordinates of the tiles of height h ≥ 1
// that must be published when publishing from a tree of
// size newTreeSize to replace a tree of size oldTreeSize.
// (No tiles need to be published for a tree of size zero.)
//
// If h ≤ 0, NewTiles panics.
func NewTiles(h int, oldTreeSize, newTreeSize int64) []Tile {
	if h <= 0 {
		panic(fmt.Sprintf("NewTiles: invalid height %d", h))
	}
	H := uint(h)
	var tiles []Tile
	for level := uint(0); newTreeSize>>(H*level) > 0; level++ {
		oldN := oldTreeSize >> (H * level)
		newN := newTreeSize >> (H * level)
		for n := oldN >> H; n < newN>>H; n++ {
			tiles = append(tiles, Tile{H: h, L: int(level), N: n, W: 1 << H})
		}
		n := newN >> H
		maxW := int(newN - n<<H)
		minW := 1
		if oldN > n<<H {
			minW = int(oldN - n<<H)
		}
		for w := minW; w <= maxW; w++ {
			tiles = append(tiles, Tile{H: h, L: int(level), N: n, W: w})
		}
	}
	return tiles
}

// ReadTileData reads the hashes for tile t from r
// and returns the corresponding tile data.
func ReadTileData(t Tile, r HashReader) ([]byte, error) {
	size := t.W
	if size == 0 {
		size = 1 << uint(t.H)
	}
	start := t.N << uint(t.H)
	indexes := make([]int64, size)
	for i := 0; i < size; i++ {
		indexes[i] = StoredHashIndex(t.H*t.L, start+int64(i))
	}

	hashes, err := r.ReadHashes(indexes)
	if err != nil {
		return nil, err
	}
	if len(hashes) != len(indexes) {
		return nil, fmt.Errorf("tlog: ReadHashes(%d indexes) = %d hashes", len(indexes), len(hashes))
	}

	tile := make([]byte, size*HashSize)
	for i := 0; i < size; i++ {
		copy(tile[i*HashSize:], hashes[i][:])
	}
	return tile, nil
}

// To limit the size of any particular directory listing,
// we encode the (possibly very large) number N
// by encoding three digits at a time.
// For example, 123456789 encodes as x123/x456/789.
// Each directory has at most 1000 each xNNN, NNN, and NNN.p children,
// so there are at most 3000 entries in any one directory.
const pathBase = 1000

// Path returns a tile coordinate path describing t.
func (t Tile) Path() string {
	n := t.N
	nStr := fmt.Sprintf("%03d", n%pathBase)
	for n >= pathBase {
		n /= pathBase
		nStr = fmt.Sprintf("x%03d/%s", n%pathBase, nStr)
	}
	pStr := ""
	if t.W != 1<<uint(t.H) {
		pStr = fmt.Sprintf(".p/%d", t.W)
	}
	var L string
	if t.L == -1 {
		L = "data"
	} else {
		L = fmt.Sprintf("%d", t.L)
	}
	return fmt.Sprintf("tile/%d/%s/%s%s", t.H, L, nStr, pStr)
}

// ParseTilePath parses a tile coordinate path.
func ParseTilePath(path string) (Tile, error) {
	f := strings.Split(path, "/")
	if len(f) < 4 || f[0] != "tile" {
		return Tile{}, &badPathError{path}
	}
	h, err1 := strconv.Atoi(f[1])
	isData := false
	if f[2] == "data" {
		isData = true
		f[2] = "0"
	}
	l, err2 := strconv.Atoi(f[2])
	if err1 != nil || err2 != nil || h < 1 || l < 0 || h > 30 {
		return Tile{}, &badPathError{path}
	}
	w := 1 << uint(h)
	if dotP := f[len(f)-2]; strings.HasSuffix(dotP, ".p") {
		ww, err := strconv.Atoi(f[len(f)-1])
		if err != nil || ww <= 0 || ww >= w {
			return Tile{}, &badPathError{path}
		}
		w = ww
		f[len(f)-2] = dotP[:len(dotP)-len(".p")]
		f = f[:len(f)-1]
	}
	f = f[3:]
	n := int64(0)
	for _, s := range f {
		nn, err := strconv.Atoi(strings.TrimPrefix(s, "x"))
		if err != nil || nn < 0 || nn >= pathBase {
			return Tile{}, &badPathError{path}
		}
		n = n*pathBase + int64(nn)
	}
	if isData {
		l = -1
	}
	t := Tile{H: h, L: l, N: n, W: w}
	if path != t.Path() {
		return Tile{}, &badPathError{path}
	}
	return t, nil
}

type badPathError struct {
	path string
}

func (e *badPathError) Error() string {
	return fmt.Sprintf("malformed tile path %q", e.path)
}

// A TileReader reads tiles from a go.sum database log.
type TileReader interface {
	// Height returns the height of the available tiles.
	Height() int

	// ReadTiles returns the data for each requested tile.
	// If ReadTiles returns err == nil, it must also return
	// a data record for each tile (len(data) == len(tiles))
	// and each data record must be the correct length
	// (len(data[i]) == tiles[i].W*HashSize).
	//
	// An implementation of ReadTiles typically reads
	// them from an on-disk cache or else from a remote
	// tile server. Tile data downloaded from a server should
	// be considered suspect and not saved into a persistent
	// on-disk cache before returning from ReadTiles.
	// When the client confirms the validity of the tile data,
	// it will call SaveTiles to signal that they can be safely
	// written to persistent storage.
	// See also https://research.swtch.com/tlog#authenticating_tiles.
	ReadTiles(tiles []Tile) (data [][]byte, err error)

	// SaveTiles informs the TileReader that the tile data
	// returned by ReadTiles has been confirmed as valid
	// and can be saved in persistent storage (on disk).
	SaveTiles(tiles []Tile, data [][]byte)
}

// TileHashReader returns a HashReader that satisfies requests
// by loading tiles of the given tree.
//
// The returned [HashReader] checks that loaded tiles are
// valid for the given tree. Therefore, any hashes returned
// by the HashReader are already proven to be in the tree.
func TileHashReader(tree Tree, tr TileReader) HashReader {
	return &tileHashReader{tree: tree, tr: tr}
}

type tileHashReader struct {
	tree Tree
	tr   TileReader
}

// tileParent returns t's k'th tile parent in the tiles for a tree of size n.
// If there is no such parent, tileParent returns Tile{}.
func tileParent(t Tile, k int, n int64) Tile {
	t.L += k
	t.N >>= uint(k * t.H)
	t.W = 1 << uint(t.H)
	if max := n >> uint(t.L*t.H); t.N<<uint(t.H)+int64(t.W) >= max {
		if t.N<<uint(t.H) >= max {
			return Tile{}
		}
		t.W = int(max - t.N<<uint(t.H))
	}
	return t
}

func (r *tileHashReader) ReadHashes(indexes []int64) ([]Hash, error) {
	h := r.tr.Height()

	tileOrder := make(map[Tile]int) // tileOrder[tileKey(tiles[i])] = i
	var tiles []Tile

	// Plan to fetch tiles necessary to recompute tree hash.
	// If it matches, those tiles are authenticated.
	stx := subTreeIndex(0, r.tree.N, nil)
	stxTileOrder := make([]int, len(stx))
	for i, x := range stx {
		tile, _, _ := tileForIndex(h, x)
		tile = tileParent(tile, 0, r.tree.N)
		if j, ok := tileOrder[tile]; ok {
			stxTileOrder[i] = j
			continue
		}
		stxTileOrder[i] = len(tiles)
		tileOrder[tile] = len(tiles)
		tiles = append(tiles, tile)
	}

	// Plan to fetch tiles containing the indexes,
	// along with any parent tiles needed
	// for authentication. For most calls,
	// the parents are being fetched anyway.
	indexTileOrder := make([]int, len(indexes))
	for i, x := range indexes {
		if x >= StoredHashIndex(0, r.tree.N) {
			return nil, fmt.Errorf("indexes not in tree")
		}

		tile, _, _ := tileForIndex(h, x)

		// Walk up parent tiles until we find one we've requested.
		// That one will be authenticated.
		k := 0
		for ; ; k++ {
			p := tileParent(tile, k, r.tree.N)
			if j, ok := tileOrder[p]; ok {
				if k == 0 {
					indexTileOrder[i] = j
				}
				break
			}
		}

		// Walk back down recording child tiles after parents.
		// This loop ends by revisiting the tile for this index
		// (tileParent(tile, 0, r.tree.N)) unless k == 0, in which
		// case the previous loop did it.
		for k--; k >= 0; k-- {
			p := tileParent(tile, k, r.tree.N)
			if p.W != 1<<uint(p.H) {
				// Only full tiles have parents.
				// This tile has a parent, so it must be full.
				return nil, fmt.Errorf("bad math in tileHashReader: %d %d %v", r.tree.N, x, p)
			}
			tileOrder[p] = len(tiles)
			if k == 0 {
				indexTileOrder[i] = len(tiles)
			}
			tiles = append(tiles, p)
		}
	}

	// Fetch all the tile data.
	data, err := r.tr.ReadTiles(tiles)
	if err != nil {
		return nil, err
	}
	if len(data) != len(tiles) {
		return nil, fmt.Errorf("TileReader returned bad result slice (len=%d, want %d)", len(data), len(tiles))
	}
	for i, tile := range tiles {
		if len(data[i]) != tile.W*HashSize {
			return nil, fmt.Errorf("TileReader returned bad result slice (%v len=%d, want %d)", tile.Path(), len(data[i]), tile.W*HashSize)
		}
	}

	// Authenticate the initial tiles against the tree hash.
	// They are arranged so that parents are authenticated before children.
	// First the tiles needed for the tree hash.
	th, err := HashFromTile(tiles[stxTileOrder[len(stx)-1]], data[stxTileOrder[len(stx)-1]], stx[len(stx)-1])
	if err != nil {
		return nil, err
	}
	for i := len(stx) - 2; i >= 0; i-- {
		h, err := HashFromTile(tiles[stxTileOrder[i]], data[stxTileOrder[i]], stx[i])
		if err != nil {
			return nil, err
		}
		th = NodeHash(h, th)
	}
	if th != r.tree.Hash {
		// The tiles do not support the tree hash.
		// We know at least one is wrong, but not which one.
		return nil, fmt.Errorf("downloaded inconsistent tile")
	}

	// Authenticate full tiles against their parents.
	for i := len(stx); i < len(tiles); i++ {
		tile := tiles[i]
		p := tileParent(tile, 1, r.tree.N)
		j, ok := tileOrder[p]
		if !ok {
			return nil, fmt.Errorf("bad math in tileHashReader %d %v: lost parent of %v", r.tree.N, indexes, tile)
		}
		h, err := HashFromTile(p, data[j], StoredHashIndex(p.L*p.H, tile.N))
		if err != nil {
			return nil, fmt.Errorf("bad math in tileHashReader %d %v: lost hash of %v: %v", r.tree.N, indexes, tile, err)
		}
		if h != tileHash(data[i]) {
			return nil, fmt.Errorf("downloaded inconsistent tile")
		}
	}

	// Now we have all the tiles needed for the requested hashes,
	// and we've authenticated the full tile set against the trusted tree hash.
	r.tr.SaveTiles(tiles, data)

	// Pull out the requested hashes.
	hashes := make([]Hash, len(indexes))
	for i, x := range indexes {
		j := indexTileOrder[i]
		h, err := HashFromTile(tiles[j], data[j], x)
		if err != nil {
			return nil, fmt.Errorf("bad math in tileHashReader %d %v: lost hash %v: %v", r.tree.N, indexes, x, err)
		}
		hashes[i] = h
	}

	return hashes, nil
}
