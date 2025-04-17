// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"

	"internal/trace/tracev2"
)

// timestamp is an unprocessed timestamp.
type timestamp uint64

// batch represents a batch of trace events.
// It is unparsed except for its header.
type batch struct {
	m    ThreadID
	time timestamp
	data []byte
	exp  tracev2.Experiment
}

func (b *batch) isStringsBatch() bool {
	return b.exp == tracev2.NoExperiment && len(b.data) > 0 && tracev2.EventType(b.data[0]) == tracev2.EvStrings
}

func (b *batch) isStacksBatch() bool {
	return b.exp == tracev2.NoExperiment && len(b.data) > 0 && tracev2.EventType(b.data[0]) == tracev2.EvStacks
}

func (b *batch) isCPUSamplesBatch() bool {
	return b.exp == tracev2.NoExperiment && len(b.data) > 0 && tracev2.EventType(b.data[0]) == tracev2.EvCPUSamples
}

func (b *batch) isFreqBatch() bool {
	return b.exp == tracev2.NoExperiment && len(b.data) > 0 && tracev2.EventType(b.data[0]) == tracev2.EvFrequency
}

// readBatch reads the next full batch from r.
func readBatch(r interface {
	io.Reader
	io.ByteReader
}) (batch, uint64, error) {
	// Read batch header byte.
	b, err := r.ReadByte()
	if err != nil {
		return batch{}, 0, err
	}
	if typ := tracev2.EventType(b); typ != tracev2.EvEventBatch && typ != tracev2.EvExperimentalBatch {
		return batch{}, 0, fmt.Errorf("expected batch event, got event %d", typ)
	}

	// Read the experiment of we have one.
	exp := tracev2.NoExperiment
	if tracev2.EventType(b) == tracev2.EvExperimentalBatch {
		e, err := r.ReadByte()
		if err != nil {
			return batch{}, 0, err
		}
		exp = tracev2.Experiment(e)
	}

	// Read the batch header: gen (generation), thread (M) ID, base timestamp
	// for the batch.
	gen, err := binary.ReadUvarint(r)
	if err != nil {
		return batch{}, gen, fmt.Errorf("error reading batch gen: %w", err)
	}
	m, err := binary.ReadUvarint(r)
	if err != nil {
		return batch{}, gen, fmt.Errorf("error reading batch M ID: %w", err)
	}
	ts, err := binary.ReadUvarint(r)
	if err != nil {
		return batch{}, gen, fmt.Errorf("error reading batch timestamp: %w", err)
	}

	// Read in the size of the batch to follow.
	size, err := binary.ReadUvarint(r)
	if err != nil {
		return batch{}, gen, fmt.Errorf("error reading batch size: %w", err)
	}
	if size > tracev2.MaxBatchSize {
		return batch{}, gen, fmt.Errorf("invalid batch size %d, maximum is %d", size, tracev2.MaxBatchSize)
	}

	// Copy out the batch for later processing.
	var data bytes.Buffer
	data.Grow(int(size))
	n, err := io.CopyN(&data, r, int64(size))
	if n != int64(size) {
		return batch{}, gen, fmt.Errorf("failed to read full batch: read %d but wanted %d", n, size)
	}
	if err != nil {
		return batch{}, gen, fmt.Errorf("copying batch data: %w", err)
	}

	// Return the batch.
	return batch{
		m:    ThreadID(m),
		time: timestamp(ts),
		data: data.Bytes(),
		exp:  exp,
	}, gen, nil
}
