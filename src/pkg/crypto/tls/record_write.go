// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"fmt";
	"hash";
	"io";
)

// writerEnableApplicationData is a message which instructs recordWriter to
// start reading and transmitting data from the application data channel.
type writerEnableApplicationData struct{}

// writerChangeCipherSpec updates the encryption and MAC functions and resets
// the sequence count.
type writerChangeCipherSpec struct {
	encryptor	encryptor;
	mac		hash.Hash;
}

// writerSetVersion sets the version number bytes that we included in the
// record header for future records.
type writerSetVersion struct {
	major, minor uint8;
}

// A recordWriter accepts messages from the handshake processor and
// application data. It writes them to the outgoing connection and blocks on
// writing. It doesn't read from the application data channel until the
// handshake processor has signaled that the handshake is complete.
type recordWriter struct {
	writer		io.Writer;
	encryptor	encryptor;
	mac		hash.Hash;
	seqNum		uint64;
	major, minor	uint8;
	shutdown	bool;
	appChan		<-chan []byte;
	controlChan	<-chan interface{};
	header		[13]byte;
}

func (w *recordWriter) loop(writer io.Writer, appChan <-chan []byte, controlChan <-chan interface{}) {
	w.writer = writer;
	w.encryptor = nop{};
	w.mac = nop{};
	w.appChan = appChan;
	w.controlChan = controlChan;

	for !w.shutdown {
		msg := <-controlChan;
		if _, ok := msg.(writerEnableApplicationData); ok {
			break;
		}
		w.processControlMessage(msg);
	}

	for !w.shutdown {
		// Always process control messages first.
		if controlMsg, ok := <-controlChan; ok {
			w.processControlMessage(controlMsg);
			continue;
		}

		select {
		case controlMsg := <-controlChan:
			w.processControlMessage(controlMsg);
		case appMsg := <-appChan:
			w.processAppMessage(appMsg);
		}
	}

	if !closed(appChan) {
		go func() {
			for _ = range appChan {
			}
		}();
	}
	if !closed(controlChan) {
		go func() {
			for _ = range controlChan {
			}
		}();
	}
}

// fillMACHeader generates a MAC header. See RFC 4346, section 6.2.3.1.
func fillMACHeader(header *[13]byte, seqNum uint64, length int, r *record) {
	header[0] = uint8(seqNum>>56);
	header[1] = uint8(seqNum>>48);
	header[2] = uint8(seqNum>>40);
	header[3] = uint8(seqNum>>32);
	header[4] = uint8(seqNum>>24);
	header[5] = uint8(seqNum>>16);
	header[6] = uint8(seqNum>>8);
	header[7] = uint8(seqNum);
	header[8] = uint8(r.contentType);
	header[9] = r.major;
	header[10] = r.minor;
	header[11] = uint8(length>>8);
	header[12] = uint8(length);
}

func (w *recordWriter) writeRecord(r *record) {
	w.mac.Reset();

	fillMACHeader(&w.header, w.seqNum, len(r.payload), r);

	w.mac.Write(w.header[0:13]);
	w.mac.Write(r.payload);
	macBytes := w.mac.Sum();

	w.encryptor.XORKeyStream(r.payload);
	w.encryptor.XORKeyStream(macBytes);

	length := len(r.payload)+len(macBytes);
	w.header[11] = uint8(length>>8);
	w.header[12] = uint8(length);
	w.writer.Write(w.header[8:13]);
	w.writer.Write(r.payload);
	w.writer.Write(macBytes);

	w.seqNum++;
}

func (w *recordWriter) processControlMessage(controlMsg interface{}) {
	if controlMsg == nil {
		w.shutdown = true;
		return;
	}

	switch msg := controlMsg.(type) {
	case writerChangeCipherSpec:
		w.writeRecord(&record{recordTypeChangeCipherSpec, w.major, w.minor, []byte{0x01}});
		w.encryptor = msg.encryptor;
		w.mac = msg.mac;
		w.seqNum = 0;
	case writerSetVersion:
		w.major = msg.major;
		w.minor = msg.minor;
	case alert:
		w.writeRecord(&record{recordTypeAlert, w.major, w.minor, []byte{byte(msg.level), byte(msg.error)}});
	case handshakeMessage:
		// TODO(agl): marshal may return a slice too large for a single record.
		w.writeRecord(&record{recordTypeHandshake, w.major, w.minor, msg.marshal()});
	default:
		fmt.Printf("processControlMessage: unknown %#v\n", msg);
	}
}

func (w *recordWriter) processAppMessage(appMsg []byte) {
	if closed(w.appChan) {
		w.writeRecord(&record{recordTypeApplicationData, w.major, w.minor, []byte{byte(alertCloseNotify)}});
		w.shutdown = true;
		return;
	}

	var done int;
	for done < len(appMsg) {
		todo := len(appMsg);
		if todo > maxTLSPlaintext {
			todo = maxTLSPlaintext;
		}
		w.writeRecord(&record{recordTypeApplicationData, w.major, w.minor, appMsg[done : done+todo]});
		done += todo;
	}
}
