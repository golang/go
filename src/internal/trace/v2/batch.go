// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"fmt"
	"io"

	"internal/trace/v2/event"
	"internal/trace/v2/event/go122"
)

// timestamp is an unprocessed timestamp.
type timestamp uint64

// batch represents a batch of trace events.
// It is unparsed except for its header.
type batch struct {
	m    ThreadID
	time timestamp
	data []byte
}

func (b *batch) isStringsBatch() bool {
	return len(b.data) > 0 && event.Type(b.data[0]) == go122.EvStrings
}

func (b *batch) isStacksBatch() bool {
	return len(b.data) > 0 && event.Type(b.data[0]) == go122.EvStacks
}

func (b *batch) isCPUSamplesBatch() bool {
	return len(b.data) > 0 && event.Type(b.data[0]) == go122.EvCPUSamples
}

func (b *batch) isFreqBatch() bool {
	return len(b.data) > 0 && event.Type(b.data[0]) == go122.EvFrequency
}

// readBatch reads the next full batch from r.
func readBatch(r *bufio.Reader) (batch, uint64, error) {
	// Read batch header byte.
	b, err := r.ReadByte()
	if err != nil {
		return batch{}, 0, err
	}
	if typ := event.Type(b); typ != go122.EvEventBatch {
		return batch{}, 0, fmt.Errorf("expected batch event (%s), got %s", go122.EventString(go122.EvEventBatch), go122.EventString(typ))
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
	if size > go122.MaxBatchSize {
		return batch{}, gen, fmt.Errorf("invalid batch size %d, maximum is %d", size, go122.MaxBatchSize)
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
	}, gen, nil
}
