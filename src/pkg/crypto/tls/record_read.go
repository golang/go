// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

// The record reader handles reading from the connection and reassembling TLS
// record structures. It loops forever doing this and writes the TLS records to
// it's outbound channel. On error, it closes its outbound channel.

import (
	"io";
	"bufio";
)

// recordReader loops, reading TLS records from source and writing them to the
// given channel. The channel is closed on EOF or on error.
func recordReader(c chan<- *record, source io.Reader) {
	defer close(c);
	buf := bufio.NewReader(source);

	for {
		var header [5]byte;
		n, _ := buf.Read(header[0:]);
		if n != 5 {
			return
		}

		recordLength := int(header[3])<<8 | int(header[4]);
		if recordLength > maxTLSCiphertext {
			return
		}

		payload := make([]byte, recordLength);
		n, _ = buf.Read(payload);
		if n != recordLength {
			return
		}

		c <- &record{recordType(header[0]), header[1], header[2], payload};
	}
}
