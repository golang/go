// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package zip

import (
	"compress/flate"
	"errors"
	"io"
	"io/ioutil"
	"sync"
)

// A Compressor returns a compressing writer, writing to the
// provided writer. On Close, any pending data should be flushed.
type Compressor func(io.Writer) (io.WriteCloser, error)

// Decompressor is a function that wraps a Reader with a decompressing Reader.
// The decompressed ReadCloser is returned to callers who open files from
// within the archive.  These callers are responsible for closing this reader
// when they're finished reading.
type Decompressor func(io.Reader) io.ReadCloser

var flateWriterPool sync.Pool

func newFlateWriter(w io.Writer) io.WriteCloser {
	fw, ok := flateWriterPool.Get().(*flate.Writer)
	if ok {
		fw.Reset(w)
	} else {
		fw, _ = flate.NewWriter(w, 5)
	}
	return &pooledFlateWriter{fw: fw}
}

type pooledFlateWriter struct {
	mu sync.Mutex // guards Close and Write
	fw *flate.Writer
}

func (w *pooledFlateWriter) Write(p []byte) (n int, err error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.fw == nil {
		return 0, errors.New("Write after Close")
	}
	return w.fw.Write(p)
}

func (w *pooledFlateWriter) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	var err error
	if w.fw != nil {
		err = w.fw.Close()
		flateWriterPool.Put(w.fw)
		w.fw = nil
	}
	return err
}

var (
	mu sync.RWMutex // guards compressor and decompressor maps

	compressors = map[uint16]Compressor{
		Store:   func(w io.Writer) (io.WriteCloser, error) { return &nopCloser{w}, nil },
		Deflate: func(w io.Writer) (io.WriteCloser, error) { return newFlateWriter(w), nil },
	}

	decompressors = map[uint16]Decompressor{
		Store:   ioutil.NopCloser,
		Deflate: flate.NewReader,
	}
)

// RegisterDecompressor allows custom decompressors for a specified method ID.
func RegisterDecompressor(method uint16, d Decompressor) {
	mu.Lock()
	defer mu.Unlock()

	if _, ok := decompressors[method]; ok {
		panic("decompressor already registered")
	}
	decompressors[method] = d
}

// RegisterCompressor registers custom compressors for a specified method ID.
// The common methods Store and Deflate are built in.
func RegisterCompressor(method uint16, comp Compressor) {
	mu.Lock()
	defer mu.Unlock()

	if _, ok := compressors[method]; ok {
		panic("compressor already registered")
	}
	compressors[method] = comp
}

func compressor(method uint16) Compressor {
	mu.RLock()
	defer mu.RUnlock()
	return compressors[method]
}

func decompressor(method uint16) Decompressor {
	mu.RLock()
	defer mu.RUnlock()
	return decompressors[method]
}
