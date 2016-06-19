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

// A Compressor returns a new compressing writer, writing to w.
// The WriteCloser's Close method must be used to flush pending data to w.
// The Compressor itself must be safe to invoke from multiple goroutines
// simultaneously, but each returned writer will be used only by
// one goroutine at a time.
type Compressor func(w io.Writer) (io.WriteCloser, error)

// A Decompressor returns a new decompressing reader, reading from r.
// The ReadCloser's Close method must be used to release associated resources.
// The Decompressor itself must be safe to invoke from multiple goroutines
// simultaneously, but each returned reader will be used only by
// one goroutine at a time.
type Decompressor func(r io.Reader) io.ReadCloser

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

var flateReaderPool sync.Pool

func newFlateReader(r io.Reader) io.ReadCloser {
	fr, ok := flateReaderPool.Get().(io.ReadCloser)
	if ok {
		fr.(flate.Resetter).Reset(r, nil)
	} else {
		fr = flate.NewReader(r)
	}
	return &pooledFlateReader{fr: fr}
}

type pooledFlateReader struct {
	mu sync.Mutex // guards Close and Read
	fr io.ReadCloser
}

func (r *pooledFlateReader) Read(p []byte) (n int, err error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.fr == nil {
		return 0, errors.New("Read after Close")
	}
	return r.fr.Read(p)
}

func (r *pooledFlateReader) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	var err error
	if r.fr != nil {
		err = r.fr.Close()
		flateReaderPool.Put(r.fr)
		r.fr = nil
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
		Deflate: newFlateReader,
	}
)

// RegisterDecompressor allows custom decompressors for a specified method ID.
// The common methods Store and Deflate are built in.
func RegisterDecompressor(method uint16, dcomp Decompressor) {
	mu.Lock()
	defer mu.Unlock()

	if _, ok := decompressors[method]; ok {
		panic("decompressor already registered")
	}
	decompressors[method] = dcomp
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
