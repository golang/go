// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package quic implements the QUIC protocol.
//
// This package is a work in progress.
// It is not ready for production usage.
// Its API is subject to change without notice.
//
// This package is low-level.
// Most users will use it indirectly through an HTTP/3 implementation.
//
// # Usage
//
// An [Endpoint] sends and receives traffic on a network address.
// Create an Endpoint to either accept inbound QUIC connections
// or create outbound ones.
//
// A [Conn] is a QUIC connection.
//
// A [Stream] is a QUIC stream, an ordered, reliable byte stream.
//
// # Cancellation
//
// All blocking operations may be canceled using a context.Context.
// When performing an operation with a canceled context, the operation
// will succeed if doing so does not require blocking. For example,
// reading from a stream will return data when buffered data is available,
// even if the stream context is canceled.
//
// # Limitations
//
// This package is a work in progress.
// Known limitations include:
//
//   - Performance is untuned.
//   - 0-RTT is not supported.
//   - Address migration is not supported.
//   - Server preferred addresses are not supported.
//   - The latency spin bit is not supported.
//   - Stream send/receive windows are configurable,
//     but are fixed and do not adapt to available throughput.
//   - Path MTU discovery is not implemented.
//
// # Security Policy
//
// This package is a work in progress,
// and not yet covered by the Go security policy.
package quic
