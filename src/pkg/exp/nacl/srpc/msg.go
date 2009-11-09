// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// SRPC constants, data structures, and parsing.

package srpc

import (
	"bytes";
	"math";
	"os";
	"strconv";
	"syscall";
	"unsafe";
)

// An Errno is an SRPC status code.
type Errno uint32

const (
	OK	Errno	= 256+iota;
	ErrBreak;
	ErrMessageTruncated;
	ErrNoMemory;
	ErrProtocolMismatch;
	ErrBadRPCNumber;
	ErrBadArgType;
	ErrTooFewArgs;
	ErrTooManyArgs;
	ErrInArgTypeMismatch;
	ErrOutArgTypeMismatch;
	ErrInternalError;
	ErrAppError;
)

var errstr = [...]string{
	OK-OK: "ok",
	ErrBreak-OK: "break",
	ErrMessageTruncated - OK: "message truncated",
	ErrNoMemory - OK: "out of memory",
	ErrProtocolMismatch - OK: "protocol mismatch",
	ErrBadRPCNumber - OK: "invalid RPC method number",
	ErrBadArgType - OK: "unexpected argument type",
	ErrTooFewArgs - OK: "too few arguments",
	ErrTooManyArgs - OK: "too many arguments",
	ErrInArgTypeMismatch - OK: "input argument type mismatch",
	ErrOutArgTypeMismatch - OK: "output argument type mismatch",
	ErrInternalError - OK: "internal error",
	ErrAppError - OK: "application error",
}

func (e Errno) String() string {
	if e < OK || int(e-OK) >= len(errstr) {
		return "Errno(" + strconv.Itoa64(int64(e)) + ")"
	}
	return errstr[e-OK];
}

// A *msgHdr is the data argument to the imc_recvmsg
// and imc_sendmsg system calls.  Because it contains unchecked
// counts trusted by the system calls, the data structure is unsafe
// to expose to package clients.
type msgHdr struct {
	iov	*iov;
	niov	int32;
	desc	*int32;
	ndesc	int32;
	flags	uint32;
}

// A single region for I/O.  Just as unsafe as msgHdr.
type iov struct {
	base	*byte;
	len	int32;
}

// A msg is the Go representation of a message.
type msg struct {
	rdata	[]byte;		// data being consumed during message parsing
	rdesc	[]int32;	// file descriptors being consumed during message parsing
	wdata	[]byte;		// data being generated when replying

	// parsed version of message
	protocol	uint32;
	requestId	uint64;
	isReq		bool;
	rpcNumber	uint32;
	gotHeader	bool;
	status		Errno;		// error code sent in response
	Arg		[]interface{};	// method arguments
	Ret		[]interface{};	// method results
	Size		[]int;		// max sizes for arrays in method results
	fmt		string;		// accumulated format string of arg+":"+ret
}

// A msgReceiver receives messages from a file descriptor.
type msgReceiver struct {
	fd	int;
	data	[128*1024]byte;
	desc	[8]int32;
	hdr	msgHdr;
	iov	iov;
}

func (r *msgReceiver) recv() (*msg, os.Error) {
	// Init pointers to buffers where syscall recvmsg can write.
	r.iov.base = &r.data[0];
	r.iov.len = int32(len(r.data));
	r.hdr.iov = &r.iov;
	r.hdr.niov = 1;
	r.hdr.desc = &r.desc[0];
	r.hdr.ndesc = int32(len(r.desc));
	n, _, e := syscall.Syscall(syscall.SYS_IMC_RECVMSG, uintptr(r.fd), uintptr(unsafe.Pointer(&r.hdr)), 0);
	if e != 0 {
		return nil, os.NewSyscallError("imc_recvmsg", int(e))
	}

	// Make a copy of the data so that the next recvmsg doesn't
	// smash it.  The system call did not update r.iov.len.  Instead it
	// returned the total byte count as n.
	m := new(msg);
	m.rdata = make([]byte, n);
	bytes.Copy(m.rdata, &r.data);

	// Make a copy of the desc too.
	// The system call *did* update r.hdr.ndesc.
	if r.hdr.ndesc > 0 {
		m.rdesc = make([]int32, r.hdr.ndesc);
		for i := range m.rdesc {
			m.rdesc[i] = r.desc[i]
		}
	}

	return m, nil;
}

// A msgSender sends messages on a file descriptor.
type msgSender struct {
	fd	int;
	hdr	msgHdr;
	iov	iov;
}

func (s *msgSender) send(m *msg) os.Error {
	if len(m.wdata) > 0 {
		s.iov.base = &m.wdata[0]
	}
	s.iov.len = int32(len(m.wdata));
	s.hdr.iov = &s.iov;
	s.hdr.niov = 1;
	s.hdr.desc = nil;
	s.hdr.ndesc = 0;
	_, _, e := syscall.Syscall(syscall.SYS_IMC_SENDMSG, uintptr(s.fd), uintptr(unsafe.Pointer(&s.hdr)), 0);
	if e != 0 {
		return os.NewSyscallError("imc_sendmsg", int(e))
	}
	return nil;
}

// Reading from msg.rdata.
func (m *msg) uint8() uint8 {
	if m.status != OK {
		return 0
	}
	if len(m.rdata) < 1 {
		m.status = ErrMessageTruncated;
		return 0;
	}
	x := m.rdata[0];
	m.rdata = m.rdata[1:len(m.rdata)];
	return x;
}

func (m *msg) uint32() uint32 {
	if m.status != OK {
		return 0
	}
	if len(m.rdata) < 4 {
		m.status = ErrMessageTruncated;
		return 0;
	}
	b := m.rdata[0:4];
	x := uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24;
	m.rdata = m.rdata[4:len(m.rdata)];
	return x;
}

func (m *msg) uint64() uint64 {
	if m.status != OK {
		return 0
	}
	if len(m.rdata) < 8 {
		m.status = ErrMessageTruncated;
		return 0;
	}
	b := m.rdata[0:8];
	x := uint64(uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24);
	x |= uint64(uint32(b[4]) | uint32(b[5])<<8 | uint32(b[6])<<16 | uint32(b[7])<<24)<<32;
	m.rdata = m.rdata[8:len(m.rdata)];
	return x;
}

func (m *msg) bytes(n int) []byte {
	if m.status != OK {
		return nil
	}
	if len(m.rdata) < n {
		m.status = ErrMessageTruncated;
		return nil;
	}
	x := m.rdata[0:n];
	m.rdata = m.rdata[n:len(m.rdata)];
	return x;
}

// Writing to msg.wdata.
func (m *msg) grow(n int) []byte {
	i := len(m.wdata);
	if i+n > cap(m.wdata) {
		a := make([]byte, i, (i+n)*2);
		bytes.Copy(a, m.wdata);
		m.wdata = a;
	}
	m.wdata = m.wdata[0 : i+n];
	return m.wdata[i : i+n];
}

func (m *msg) wuint8(x uint8)	{ m.grow(1)[0] = x }

func (m *msg) wuint32(x uint32) {
	b := m.grow(4);
	b[0] = byte(x);
	b[1] = byte(x>>8);
	b[2] = byte(x>>16);
	b[3] = byte(x>>24);
}

func (m *msg) wuint64(x uint64) {
	b := m.grow(8);
	lo := uint32(x);
	b[0] = byte(lo);
	b[1] = byte(lo>>8);
	b[2] = byte(lo>>16);
	b[3] = byte(lo>>24);
	hi := uint32(x>>32);
	b[4] = byte(hi);
	b[5] = byte(hi>>8);
	b[6] = byte(hi>>16);
	b[7] = byte(hi>>24);
}

func (m *msg) wbytes(p []byte)	{ bytes.Copy(m.grow(len(p)), p) }

func (m *msg) wstring(s string) {
	b := m.grow(len(s));
	for i := range b {
		b[i] = s[i]
	}
}

// Parsing of RPC header and arguments.
//
// The header format is:
//	protocol uint32;
//	requestId uint64;
//	isReq bool;
//	rpcNumber uint32;
//	status uint32;  // only for response
//
// Then a sequence of values follow, preceded by the length:
//	nvalue uint32;
//
// Each value begins with a one-byte type followed by
// type-specific data.
//
//	type uint8;
//	'b':	x bool;
//	'C':	len uint32; x [len]byte;
//	'd':	x float64;
//	'D':	len uint32; x [len]float64;
//	'h':	x int;	// handle aka file descriptor
//	'i':	x int32;
//	'I':	len uint32; x [len]int32;
//	's':	len uint32; x [len]byte;
//
// If this is a request, a sequence of pseudo-values follows,
// preceded by its length (nvalue uint32).
//
// Each pseudo-value is a one-byte type as above,
// followed by a maximum length (len uint32)
// for the 'C', 'D', 'I', and 's' types.
//
// In the Go msg, we represent each argument by
// an empty interface containing the type of x in the
// corresponding case.

// The current protocol number.
const protocol = 0xc0da0002

func (m *msg) unpackHeader() {
	m.protocol = m.uint32();
	m.requestId = m.uint64();
	m.isReq = m.uint8() != 0;
	m.rpcNumber = m.uint32();
	m.gotHeader = m.status == OK;	// signal that header parsed successfully
	if m.gotHeader && !m.isReq {
		status := Errno(m.uint32());
		m.gotHeader = m.status == OK;	// still ok?
		if m.gotHeader {
			m.status = status
		}
	}
}

func (m *msg) packHeader() {
	m.wuint32(m.protocol);
	m.wuint64(m.requestId);
	if m.isReq {
		m.wuint8(1)
	} else {
		m.wuint8(0)
	}
	m.wuint32(m.rpcNumber);
	if !m.isReq {
		m.wuint32(uint32(m.status))
	}
}

func (m *msg) unpackValues(v []interface{}) {
	for i := range v {
		t := m.uint8();
		m.fmt += string(t);
		switch t {
		default:
			if m.status == OK {
				m.status = ErrBadArgType
			}
			return;
		case 'b':	// bool[1]
			v[i] = m.uint8() > 0
		case 'C':	// char array
			v[i] = m.bytes(int(m.uint32()))
		case 'd':	// double
			v[i] = math.Float64frombits(m.uint64())
		case 'D':	// double array
			a := make([]float64, int(m.uint32()));
			for j := range a {
				a[j] = math.Float64frombits(m.uint64())
			}
			v[i] = a;
		case 'h':	// file descriptor (handle)
			if len(m.rdesc) == 0 {
				if m.status == OK {
					m.status = ErrBadArgType
				}
				return;
			}
			v[i] = int(m.rdesc[0]);
			m.rdesc = m.rdesc[1:len(m.rdesc)];
		case 'i':	// int
			v[i] = int32(m.uint32())
		case 'I':	// int array
			a := make([]int32, int(m.uint32()));
			for j := range a {
				a[j] = int32(m.uint32())
			}
			v[i] = a;
		case 's':	// string
			v[i] = string(m.bytes(int(m.uint32())))
		}
	}
}

func (m *msg) packValues(v []interface{}) {
	for i := range v {
		switch x := v[i].(type) {
		default:
			if m.status == OK {
				m.status = ErrInternalError
			}
			return;
		case bool:
			m.wuint8('b');
			if x {
				m.wuint8(1)
			} else {
				m.wuint8(0)
			}
		case []byte:
			m.wuint8('C');
			m.wuint32(uint32(len(x)));
			m.wbytes(x);
		case float64:
			m.wuint8('d');
			m.wuint64(math.Float64bits(x));
		case []float64:
			m.wuint8('D');
			m.wuint32(uint32(len(x)));
			for _, f := range x {
				m.wuint64(math.Float64bits(f))
			}
		case int32:
			m.wuint8('i');
			m.wuint32(uint32(x));
		case []int32:
			m.wuint8('I');
			m.wuint32(uint32(len(x)));
			for _, i := range x {
				m.wuint32(uint32(i))
			}
		case string:
			m.wuint8('s');
			m.wuint32(uint32(len(x)));
			m.wstring(x);
		}
	}
}

func (m *msg) unpackRequest() {
	m.status = OK;
	if m.unpackHeader(); m.status != OK {
		return
	}
	if m.protocol != protocol || !m.isReq {
		m.status = ErrProtocolMismatch;
		return;
	}

	// type-tagged argument values
	m.Arg = make([]interface{}, m.uint32());
	m.unpackValues(m.Arg);
	if m.status != OK {
		return
	}

	// type-tagged expected return sizes.
	// fill in zero values for each return value
	// and save sizes.
	m.fmt += ":";
	m.Ret = make([]interface{}, m.uint32());
	m.Size = make([]int, len(m.Ret));
	for i := range m.Ret {
		t := m.uint8();
		m.fmt += string(t);
		switch t {
		default:
			if m.status == OK {
				m.status = ErrBadArgType
			}
			return;
		case 'b':	// bool[1]
			m.Ret[i] = false
		case 'C':	// char array
			m.Size[i] = int(m.uint32());
			m.Ret[i] = []byte(nil);
		case 'd':	// double
			m.Ret[i] = float64(0)
		case 'D':	// double array
			m.Size[i] = int(m.uint32());
			m.Ret[i] = []float64(nil);
		case 'h':	// file descriptor (handle)
			m.Ret[i] = int(-1)
		case 'i':	// int
			m.Ret[i] = int32(0)
		case 'I':	// int array
			m.Size[i] = int(m.uint32());
			m.Ret[i] = []int32(nil);
		case 's':	// string
			m.Size[i] = int(m.uint32());
			m.Ret[i] = "";
		}
	}
}

func (m *msg) packRequest() {
	m.packHeader();
	m.wuint32(uint32(len(m.Arg)));
	m.packValues(m.Arg);
	m.wuint32(uint32(len(m.Ret)));
	for i, v := range m.Ret {
		switch x := v.(type) {
		case bool:
			m.wuint8('b')
		case []byte:
			m.wuint8('C');
			m.wuint32(uint32(m.Size[i]));
		case float64:
			m.wuint8('d')
		case []float64:
			m.wuint8('D');
			m.wuint32(uint32(m.Size[i]));
		case int:
			m.wuint8('h')
		case int32:
			m.wuint8('i')
		case []int32:
			m.wuint8('I');
			m.wuint32(uint32(m.Size[i]));
		case string:
			m.wuint8('s');
			m.wuint32(uint32(m.Size[i]));
		}
	}
}

func (m *msg) unpackResponse() {
	m.status = OK;
	if m.unpackHeader(); m.status != OK {
		return
	}
	if m.protocol != protocol || m.isReq {
		m.status = ErrProtocolMismatch;
		return;
	}

	// type-tagged return values
	m.fmt = "";
	m.Ret = make([]interface{}, m.uint32());
	m.unpackValues(m.Ret);
}

func (m *msg) packResponse() {
	m.packHeader();
	m.wuint32(uint32(len(m.Ret)));
	m.packValues(m.Ret);
}
