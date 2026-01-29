// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !nethttpomithttp2

package http

import (
	"errors"
	"fmt"
	"testing"
)

type externalStreamErrorCode uint32

type externalStreamError struct {
	StreamID uint32
	Code     externalStreamErrorCode
	Cause    error
}

func (e externalStreamError) Error() string {
	return fmt.Sprintf("ID %v, code %v", e.StreamID, e.Code)
}

func TestStreamError(t *testing.T) {
	streamErr := http2streamError(42, http2ErrCodeProtocol)
	extStreamErr, ok := errors.AsType[externalStreamError](streamErr)
	if !ok {
		t.Fatalf("errors.AsType failed")
	}
	if extStreamErr.StreamID != streamErr.StreamID {
		t.Errorf("got StreamID %v, expected %v", extStreamErr.StreamID, streamErr.StreamID)
	}
	if extStreamErr.Cause != streamErr.Cause {
		t.Errorf("got Cause %v, expected %v", extStreamErr.Cause, streamErr.Cause)
	}
	if uint32(extStreamErr.Code) != uint32(streamErr.Code) {
		t.Errorf("got Code %v, expected %v", extStreamErr.Code, streamErr.Code)
	}
}
