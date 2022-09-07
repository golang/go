// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !nethttpomithttp2
// +build !nethttpomithttp2

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
	var target externalStreamError
	streamErr := http2streamError(42, http2ErrCodeProtocol)
	ok := errors.As(streamErr, &target)
	if !ok {
		t.Fatalf("errors.As failed")
	}
	if target.StreamID != streamErr.StreamID {
		t.Errorf("got StreamID %v, expected %v", target.StreamID, streamErr.StreamID)
	}
	if target.Cause != streamErr.Cause {
		t.Errorf("got Cause %v, expected %v", target.Cause, streamErr.Cause)
	}
	if uint32(target.Code) != uint32(streamErr.Code) {
		t.Errorf("got Code %v, expected %v", target.Code, streamErr.Code)
	}
}
