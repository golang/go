// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package httpcommon

import (
	"compress/flate"
	"compress/gzip"
	"errors"
	"io"
	"io/fs"
	"sync"
)

var errConcurrentRead = errors.New("http: concurrent read on response body")

// incomparable is a zero-width, non-comparable type. Adding it to a struct
// makes that struct also non-comparable, and generally doesn't add
// any size (as long as it's first).
type incomparable [0]func()

// GzipReader wraps a response body so it can lazily
// get gzip.Reader from the pool on the first call to Read.
// After Close is called it puts gzip.Reader to the pool immediately
// if there is no Read in progress or later when Read completes.
type GzipReader struct {
	_    incomparable
	Body io.ReadCloser // underlying Response.Body
	mu   sync.Mutex    // guards zr and zerr
	zr   *gzip.Reader  // stores gzip reader from the pool between reads
	zerr error         // sticky gzip reader init error or sentinel value to detect concurrent read and read after close
}

type eofReader struct{}

func (eofReader) Read([]byte) (int, error) { return 0, io.EOF }
func (eofReader) ReadByte() (byte, error)  { return 0, io.EOF }

var gzipPool = sync.Pool{New: func() any { return new(gzip.Reader) }}

// gzipPoolGet gets a gzip.Reader from the pool and resets it to read from r.
func gzipPoolGet(r io.Reader) (*gzip.Reader, error) {
	zr := gzipPool.Get().(*gzip.Reader)
	if err := zr.Reset(r); err != nil {
		gzipPoolPut(zr)
		return nil, err
	}
	return zr, nil
}

// gzipPoolPut puts a gzip.Reader back into the pool.
func gzipPoolPut(zr *gzip.Reader) {
	// Reset will allocate bufio.Reader if we pass it anything
	// other than a flate.Reader, so ensure that it's getting one.
	var r flate.Reader = eofReader{}
	zr.Reset(r)
	gzipPool.Put(zr)
}

// acquire returns a gzip.Reader for reading response body.
// The reader must be released after use.
func (gz *GzipReader) acquire() (*gzip.Reader, error) {
	gz.mu.Lock()
	defer gz.mu.Unlock()
	if gz.zerr != nil {
		return nil, gz.zerr
	}
	if gz.zr == nil {
		// gzipPoolGet might block indefinitely since it reads the gzip header.
		// Therefore, drop mu temporarily when using gzipPoolGet.
		// We set zerr to errConcurrentRead to prevent concurrent read
		// even when mu is temporarily dropped.
		gz.zerr = errConcurrentRead
		gz.mu.Unlock()
		zr, err := gzipPoolGet(gz.Body)
		gz.mu.Lock()
		// Guard against Close being called while gzipPoolGet is running.
		if gz.zerr != errConcurrentRead {
			if zr != nil {
				gzipPoolPut(zr)
			}
			return nil, gz.zerr
		}
		gz.zr, gz.zerr = zr, err
		if gz.zerr != nil {
			return nil, gz.zerr
		}
	}
	ret := gz.zr
	gz.zr, gz.zerr = nil, errConcurrentRead
	return ret, nil
}

// release returns the gzip.Reader to the pool if Close was called during Read.
func (gz *GzipReader) release(zr *gzip.Reader) {
	gz.mu.Lock()
	defer gz.mu.Unlock()
	if gz.zerr == errConcurrentRead {
		gz.zr, gz.zerr = zr, nil
	} else { // fs.ErrClosed
		gzipPoolPut(zr)
	}
}

// close returns the gzip.Reader to the pool immediately or
// signals release to do so after Read completes.
func (gz *GzipReader) close() {
	gz.mu.Lock()
	defer gz.mu.Unlock()
	if gz.zerr == nil && gz.zr != nil {
		gzipPoolPut(gz.zr)
		gz.zr = nil
	}
	gz.zerr = fs.ErrClosed
}

func (gz *GzipReader) Read(p []byte) (n int, err error) {
	zr, err := gz.acquire()
	if err != nil {
		return 0, err
	}
	defer gz.release(zr)

	return zr.Read(p)
}

func (gz *GzipReader) Close() error {
	gz.close()

	return gz.Body.Close()
}
