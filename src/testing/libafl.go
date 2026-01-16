// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"sync"
	"syscall"
	"unsafe"
)

type libaflHarness struct {
	once sync.Once

	mu          sync.Mutex
	captureMode bool
	captured    bool
	capturedFn  reflect.Value
	capturedTyp []reflect.Type
	seeds       [][]byte

	fn       reflect.Value
	argTypes []reflect.Type
	name     string
	initErr  error
}

var libafl libaflHarness

const (
	libaflMaxVarLen       = 1 << 20
	libaflMaxContainerLen = 4096
	libaflMaxDepth        = 32
)

// LibAFLInit initializes cybergo's LibAFL fuzz harness for the given fuzz test.
// This is used by the generated test main when 'go test -fuzz ... -use-libafl'
// is set.
func LibAFLInit(name string, fuzzFn func(*F)) error {
	libafl.once.Do(func() {
		// In --use-libafl mode we don't run the generated test main, so the
		// testing flags are not registered unless we do it ourselves.
		// Without this, methods like (*T).Fatalf may crash (e.g. nil *fullPath).
		Init()

		libafl.mu.Lock()
		libafl.captureMode = true
		libafl.captured = false
		libafl.capturedFn = reflect.Value{}
		libafl.capturedTyp = nil
		libafl.seeds = nil
		libafl.name = name
		libafl.mu.Unlock()

		f := F{}
		f.name = name
		done := make(chan struct{})
		go func() {
			defer close(done)
			fuzzFn(&f)
		}()
		<-done

		libafl.mu.Lock()
		defer libafl.mu.Unlock()
		libafl.captureMode = false

		if !libafl.captured || !libafl.capturedFn.IsValid() {
			libafl.initErr = errors.New("fuzz target did not call F.Fuzz")
			return
		}
		if len(libafl.capturedTyp) == 0 {
			libafl.initErr = errors.New("unsupported libafl fuzz signature: fuzz target has no parameters")
			return
		}

		types := slices.Clone(libafl.capturedTyp)
		for _, c := range f.corpus {
			if len(c.Values) != len(types) {
				libafl.initErr = fmt.Errorf("wrong number of values in corpus entry: %d, want %d", len(c.Values), len(types))
				return
			}
			valsT := make([]reflect.Type, len(c.Values))
			for i, v := range c.Values {
				valsT[i] = reflect.TypeOf(v)
			}
			for i := range types {
				if valsT[i] != types[i] {
					libafl.initErr = fmt.Errorf("mismatched types in corpus entry: %v, want %v", valsT, types)
					return
				}
			}

			seed, ok := libaflMarshalInputs(c.Values, types)
			if !ok {
				libafl.initErr = errors.New("failed to marshal libafl corpus entry")
				return
			}
			libafl.seeds = append(libafl.seeds, seed)
		}
		libafl.fn = libafl.capturedFn
		libafl.argTypes = types
	})

	return libafl.initErr
}

// LibAFLWriteSeeds writes the captured f.Add corpus entries to dir.
//
// This is used to seed LibAFL's initial on-disk corpus directory, so the fuzzer
// doesn't have to start from purely random bytes.
func LibAFLWriteSeeds(dir string) error {
	if dir == "" {
		return nil
	}

	libafl.mu.Lock()
	seeds := slices.Clone(libafl.seeds)
	initErr := libafl.initErr
	libafl.mu.Unlock()

	if initErr != nil {
		return initErr
	}
	if len(seeds) == 0 {
		return nil
	}

	if err := os.MkdirAll(dir, 0777); err != nil {
		return err
	}

	for i, seed := range seeds {
		path := filepath.Join(dir, fmt.Sprintf("cybergo-add-seed-%d", i))
		f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0666)
		if err != nil {
			if errors.Is(err, fs.ErrExist) {
				continue
			}
			return err
		}
		_, werr := f.Write(seed)
		cerr := f.Close()
		if werr != nil {
			return werr
		}
		if cerr != nil {
			return cerr
		}
	}
	return nil
}

func libaflCapture(fn reflect.Value, types []reflect.Type) bool {
	libafl.mu.Lock()
	defer libafl.mu.Unlock()
	if !libafl.captureMode || libafl.captured {
		return false
	}
	libafl.captured = true
	libafl.capturedFn = fn
	libafl.capturedTyp = slices.Clip(types)
	return true
}

// LibAFLFuzzOneInput runs the captured fuzz target on a single input.
// It crashes the process if the fuzz target reports a failure.
func LibAFLFuzzOneInput(data []byte) {
	libafl.mu.Lock()
	fn := libafl.fn
	argTypes := libafl.argTypes
	name := libafl.name
	initErr := libafl.initErr
	libafl.mu.Unlock()

	if initErr != nil {
		panic(initErr)
	}
	if !fn.IsValid() {
		panic("libafl fuzz target not initialized")
	}

	// Run the fuzz target in a test goroutine so that t.Fatal/t.FailNow
	// (which call runtime.Goexit) don't unwind the cgo callback goroutine.
	tstate := newTestState(1, allMatcher())
	tstate.isFuzzing = true
	root := common{w: os.Stdout}

	ctx, cancelCtx := context.WithCancel(context.Background())
	t := &T{
		common: common{
			barrier:   make(chan bool),
			signal:    make(chan bool, 1),
			name:      name,
			parent:    &root,
			level:     root.level + 1,
			chatty:    nil,
			ctx:       ctx,
			cancelCtx: cancelCtx,
		},
		tstate: tstate,
	}
	t.w = indenter{&t.common}
	t.setOutputWriter()

	go tRunner(t, func(t *T) {
		args := libaflUnmarshalArgs(data, argTypes)
		fn.Call(append([]reflect.Value{reflect.ValueOf(t)}, args...))
	})
	<-t.signal

	if t.Failed() {
		// Signal a crash to the LibAFL in-process executor.
		_ = syscall.Kill(os.Getpid(), syscall.SIGABRT)
		panic("fuzz target failed")
	}
}

func libaflMarshalInputs(values []any, types []reflect.Type) ([]byte, bool) {
	if len(values) != len(types) {
		return nil, false
	}
	for i := range types {
		if reflect.TypeOf(values[i]) != types[i] {
			return nil, false
		}
	}

	// Preserve the simplest, most ergonomic mapping for one-arg fuzz targets:
	// the fuzzer's input bytes are passed through verbatim as []byte or string.
	if len(types) == 1 {
		switch types[0] {
		case reflect.TypeFor[[]byte]():
			return slices.Clone(values[0].([]byte)), true
		case reflect.TypeFor[string]():
			return []byte(values[0].(string)), true
		}
	}

	var b []byte
	var scratch [binary.MaxVarintLen64]byte
	for i, t := range types {
		var ok bool
		b, ok = libaflAppendValue(b, reflect.ValueOf(values[i]), t, &scratch, 0)
		if !ok {
			return nil, false
		}
	}
	return b, true
}

func libaflAppendValue(dst []byte, v reflect.Value, t reflect.Type, scratch *[binary.MaxVarintLen64]byte, depth int) ([]byte, bool) {
	if depth > libaflMaxDepth {
		return dst, true
	}
	if !v.IsValid() {
		return nil, false
	}
	if v.Type() != t {
		return nil, false
	}

	switch t.Kind() {
	case reflect.Bool:
		if v.Bool() {
			return append(dst, 1), true
		}
		return append(dst, 0), true

	case reflect.Int:
		var tmp [8]byte
		binary.LittleEndian.PutUint64(tmp[:], uint64(v.Int()))
		return append(dst, tmp[:]...), true
	case reflect.Int8:
		return append(dst, byte(v.Int())), true
	case reflect.Int16:
		var tmp [2]byte
		binary.LittleEndian.PutUint16(tmp[:], uint16(v.Int()))
		return append(dst, tmp[:]...), true
	case reflect.Int32:
		var tmp [4]byte
		binary.LittleEndian.PutUint32(tmp[:], uint32(v.Int()))
		return append(dst, tmp[:]...), true
	case reflect.Int64:
		var tmp [8]byte
		binary.LittleEndian.PutUint64(tmp[:], uint64(v.Int()))
		return append(dst, tmp[:]...), true

	case reflect.Uint:
		var tmp [8]byte
		binary.LittleEndian.PutUint64(tmp[:], v.Uint())
		return append(dst, tmp[:]...), true
	case reflect.Uint8:
		return append(dst, byte(v.Uint())), true
	case reflect.Uint16:
		var tmp [2]byte
		binary.LittleEndian.PutUint16(tmp[:], uint16(v.Uint()))
		return append(dst, tmp[:]...), true
	case reflect.Uint32:
		var tmp [4]byte
		binary.LittleEndian.PutUint32(tmp[:], uint32(v.Uint()))
		return append(dst, tmp[:]...), true
	case reflect.Uint64:
		var tmp [8]byte
		binary.LittleEndian.PutUint64(tmp[:], v.Uint())
		return append(dst, tmp[:]...), true

	case reflect.Float32:
		var tmp [4]byte
		binary.LittleEndian.PutUint32(tmp[:], math.Float32bits(float32(v.Float())))
		return append(dst, tmp[:]...), true
	case reflect.Float64:
		var tmp [8]byte
		binary.LittleEndian.PutUint64(tmp[:], math.Float64bits(v.Float()))
		return append(dst, tmp[:]...), true

	case reflect.String:
		s := v.String()
		n := binary.PutUvarint(scratch[:], uint64(len(s)))
		dst = append(dst, scratch[:n]...)
		dst = append(dst, s...)
		return dst, true

	case reflect.Slice:
		if t.Elem().Kind() == reflect.Uint8 {
			n := v.Len()
			if n > libaflMaxVarLen {
				n = libaflMaxVarLen
			}
			nn := binary.PutUvarint(scratch[:], uint64(n))
			dst = append(dst, scratch[:nn]...)
			for i := 0; i < n; i++ {
				dst = append(dst, byte(v.Index(i).Uint()))
			}
			return dst, true
		}

		n := v.Len()
		if n > libaflMaxContainerLen {
			n = libaflMaxContainerLen
		}
		nn := binary.PutUvarint(scratch[:], uint64(n))
		dst = append(dst, scratch[:nn]...)
		for i := 0; i < n; i++ {
			var ok bool
			dst, ok = libaflAppendValue(dst, v.Index(i), t.Elem(), scratch, depth+1)
			if !ok {
				return nil, false
			}
		}
		return dst, true

	case reflect.Array:
		n := v.Len()
		if n > libaflMaxContainerLen {
			n = libaflMaxContainerLen
		}
		for i := 0; i < n; i++ {
			var ok bool
			dst, ok = libaflAppendValue(dst, v.Index(i), t.Elem(), scratch, depth+1)
			if !ok {
				return nil, false
			}
		}
		return dst, true

	case reflect.Struct:
		n := v.NumField()
		if n > libaflMaxContainerLen {
			n = libaflMaxContainerLen
		}
		for i := 0; i < n; i++ {
			var ok bool
			dst, ok = libaflAppendValue(dst, v.Field(i), t.Field(i).Type, scratch, depth+1)
			if !ok {
				return nil, false
			}
		}
		return dst, true

	case reflect.Ptr:
		if v.IsNil() {
			return append(dst, 0), true
		}
		dst = append(dst, 1)
		return libaflAppendValue(dst, v.Elem(), t.Elem(), scratch, depth+1)
	}

	return nil, false
}

type libaflReader struct {
	b []byte
	i int
}

func (r *libaflReader) empty() bool {
	return r.i >= len(r.b)
}

func (r *libaflReader) readByte() byte {
	if r.empty() {
		return 0
	}
	v := r.b[r.i]
	r.i++
	return v
}

func (r *libaflReader) readUvarint() uint64 {
	if r.empty() {
		return 0
	}
	v, n := binary.Uvarint(r.b[r.i:])
	if n > 0 {
		r.i += n
		return v
	}
	// Invalid varint. Consume a byte so we make progress.
	r.i++
	return 0
}

func (r *libaflReader) takeBytes(n int) []byte {
	if n <= 0 || r.empty() {
		return nil
	}
	remain := len(r.b) - r.i
	if n > remain {
		n = remain
	}
	start := r.i
	r.i += n
	// Set cap == len so append does not overwrite subsequent args.
	return r.b[start:r.i:r.i]
}

func (r *libaflReader) readUint16() uint16 {
	var tmp [2]byte
	copy(tmp[:], r.takeBytes(2))
	return binary.LittleEndian.Uint16(tmp[:])
}

func (r *libaflReader) readUint32() uint32 {
	var tmp [4]byte
	copy(tmp[:], r.takeBytes(4))
	return binary.LittleEndian.Uint32(tmp[:])
}

func (r *libaflReader) readUint64() uint64 {
	var tmp [8]byte
	copy(tmp[:], r.takeBytes(8))
	return binary.LittleEndian.Uint64(tmp[:])
}

func libaflUnmarshalArgs(data []byte, argTypes []reflect.Type) []reflect.Value {
	if len(argTypes) == 1 {
		switch argTypes[0] {
		case reflect.TypeFor[[]byte]():
			return []reflect.Value{reflect.ValueOf(data)}
		case reflect.TypeFor[string]():
			if len(data) > libaflMaxVarLen {
				data = data[:libaflMaxVarLen]
			}
			return []reflect.Value{reflect.ValueOf(string(data))}
		}
	}

	r := &libaflReader{b: data}
	args := make([]reflect.Value, 0, len(argTypes))
	for _, t := range argTypes {
		args = append(args, libaflDecodeValue(r, t, 0))
	}
	return args
}

func libaflDecodeValue(r *libaflReader, t reflect.Type, depth int) reflect.Value {
	if depth > libaflMaxDepth {
		return reflect.Zero(t)
	}

	switch t.Kind() {
	case reflect.Bool:
		v := reflect.New(t).Elem()
		v.SetBool(r.readByte()&1 == 1)
		return v

	case reflect.Int:
		v := reflect.New(t).Elem()
		v.SetInt(int64(r.readUint64()))
		return v
	case reflect.Int8:
		v := reflect.New(t).Elem()
		v.SetInt(int64(int8(r.readByte())))
		return v
	case reflect.Int16:
		v := reflect.New(t).Elem()
		v.SetInt(int64(int16(r.readUint16())))
		return v
	case reflect.Int32:
		v := reflect.New(t).Elem()
		v.SetInt(int64(int32(r.readUint32())))
		return v
	case reflect.Int64:
		v := reflect.New(t).Elem()
		v.SetInt(int64(r.readUint64()))
		return v

	case reflect.Uint:
		v := reflect.New(t).Elem()
		v.SetUint(r.readUint64())
		return v
	case reflect.Uint8:
		v := reflect.New(t).Elem()
		v.SetUint(uint64(r.readByte()))
		return v
	case reflect.Uint16:
		v := reflect.New(t).Elem()
		v.SetUint(uint64(r.readUint16()))
		return v
	case reflect.Uint32:
		v := reflect.New(t).Elem()
		v.SetUint(uint64(r.readUint32()))
		return v
	case reflect.Uint64:
		v := reflect.New(t).Elem()
		v.SetUint(r.readUint64())
		return v

	case reflect.Float32:
		v := reflect.New(t).Elem()
		v.SetFloat(float64(math.Float32frombits(r.readUint32())))
		return v
	case reflect.Float64:
		v := reflect.New(t).Elem()
		v.SetFloat(math.Float64frombits(r.readUint64()))
		return v

	case reflect.String:
		n := libaflReadLen(r, libaflMaxVarLen)
		b := r.takeBytes(n)
		v := reflect.New(t).Elem()
		v.SetString(string(b))
		return v

	case reflect.Slice:
		n := libaflReadLen(r, libaflMaxVarLen)
		if t.Elem().Kind() == reflect.Uint8 {
			b := r.takeBytes(n)
			if t.Elem() == reflect.TypeFor[uint8]() {
				return reflect.ValueOf(b).Convert(t)
			}
			out := reflect.MakeSlice(t, len(b), len(b))
			for i := range b {
				out.Index(i).SetUint(uint64(b[i]))
			}
			return out
		}
		if n > libaflMaxContainerLen {
			n = libaflMaxContainerLen
		}
		out := reflect.MakeSlice(t, n, n)
		for i := 0; i < n; i++ {
			out.Index(i).Set(libaflDecodeValue(r, t.Elem(), depth+1))
		}
		return out

	case reflect.Array:
		out := reflect.New(t).Elem()
		n := out.Len()
		if n > libaflMaxContainerLen {
			n = libaflMaxContainerLen
		}
		for i := 0; i < n; i++ {
			out.Index(i).Set(libaflDecodeValue(r, t.Elem(), depth+1))
		}
		return out

	case reflect.Struct:
		out := reflect.New(t).Elem()
		n := out.NumField()
		if n > libaflMaxContainerLen {
			n = libaflMaxContainerLen
		}
		for i := 0; i < n; i++ {
			f := out.Field(i)
			v := libaflDecodeValue(r, t.Field(i).Type, depth+1)
			if f.CanSet() {
				f.Set(v)
				continue
			}
			// Populate unexported fields too; fuzzing wants to break invariants.
			f = reflect.NewAt(f.Type(), unsafe.Pointer(f.UnsafeAddr())).Elem()
			f.Set(v)
		}
		return out

	case reflect.Ptr:
		if r.readByte()&1 == 0 {
			return reflect.Zero(t)
		}
		p := reflect.New(t.Elem())
		p.Elem().Set(libaflDecodeValue(r, t.Elem(), depth+1))
		return p
	}

	return reflect.Zero(t)
}

func libaflReadLen(r *libaflReader, max int) int {
	n64 := r.readUvarint()
	remain := len(r.b) - r.i
	if n64 > uint64(remain) {
		n64 = uint64(remain)
	}
	if n64 > uint64(max) {
		n64 = uint64(max)
	}
	return int(n64)
}
