// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Native Client SRPC message passing.
// This code is needed to invoke SecureRandom, the NaCl equivalent of /dev/random.

package syscall

import (
	"errors"
	"sync"
	"unsafe"
)

// An srpcClient represents the client side of an SRPC connection.
type srpcClient struct {
	fd      int // to server
	r       msgReceiver
	s       msgSender
	service map[string]srpcService // services by name

	outMu sync.Mutex // protects writing to connection

	mu      sync.Mutex // protects following fields
	muxer   bool       // is someone reading and muxing responses
	pending map[uint32]*srpc
	idGen   uint32 // generator for request IDs
}

// An srpcService is a single method that the server offers.
type srpcService struct {
	num uint32 // method number
	fmt string // argument format; see "parsing of RPC messages" below
}

// An srpc represents a single srpc issued by a client.
type srpc struct {
	Ret  []interface{}
	Done chan *srpc
	Err  error
	c    *srpcClient
	id   uint32
}

// newClient allocates a new SRPC client using the file descriptor fd.
func newClient(fd int) (*srpcClient, error) {
	c := new(srpcClient)
	c.fd = fd
	c.r.fd = fd
	c.s.fd = fd
	c.service = make(map[string]srpcService)
	c.pending = make(map[uint32]*srpc)

	// service discovery request
	m := &msg{
		isRequest: 1,
		template:  []interface{}{[]byte(nil)},
		size:      []int{4000}, // max size to accept for returned byte slice
	}
	if err := m.pack(); err != nil {
		return nil, errors.New("Native Client SRPC service_discovery: preparing request: " + err.Error())
	}
	c.s.send(m)
	m, err := c.r.recv()
	if err != nil {
		return nil, err
	}
	m.unpack()
	if m.status != uint32(srpcOK) {
		return nil, errors.New("Native Client SRPC service_discovery: " + srpcErrno(m.status).Error())
	}
	list := m.value[0].([]byte)
	var n uint32
	for len(list) > 0 {
		var line []byte
		i := byteIndex(list, '\n')
		if i < 0 {
			line, list = list, nil
		} else {
			line, list = list[:i], list[i+1:]
		}
		i = byteIndex(line, ':')
		if i >= 0 {
			c.service[string(line)] = srpcService{n, string(line[i+1:])}
		}
		n++
	}

	return c, nil
}

func byteIndex(b []byte, c byte) int {
	for i, bi := range b {
		if bi == c {
			return i
		}
	}
	return -1
}

var yourTurn srpc

func (c *srpcClient) wait(r *srpc) {
	var rx *srpc
	for rx = range r.Done {
		if rx != &yourTurn {
			break
		}
		c.input()
	}
	return
}

func (c *srpcClient) input() {
	// read message
	m, err := c.r.recv()
	if err != nil {
		println("Native Client SRPC receive error:", err.Error())
		return
	}
	if m.unpack(); m.status != uint32(srpcOK) {
		println("Native Client SRPC receive error: invalid message: ", srpcErrno(m.status).Error())
		return
	}

	// deliver to intended recipient
	c.mu.Lock()
	rpc, ok := c.pending[m.id]
	if ok {
		delete(c.pending, m.id)
	}

	// wake a new muxer if there are more RPCs to read
	c.muxer = false
	for _, rpc := range c.pending {
		c.muxer = true
		rpc.Done <- &yourTurn
		break
	}
	c.mu.Unlock()
	if !ok {
		println("Native Client: unexpected response for ID", m.id)
		return
	}
	rpc.Ret = m.value
	rpc.Done <- rpc
}

// Wait blocks until the RPC has finished.
func (r *srpc) Wait() {
	r.c.wait(r)
}

// Start issues an RPC request for method name with the given arguments.
// The RPC r must not be in use for another pending request.
// To wait for the RPC to finish, receive from r.Done and then
// inspect r.Ret and r.Errno.
func (r *srpc) Start(name string, arg []interface{}) {
	r.Err = nil
	r.c.mu.Lock()
	srv, ok := r.c.service[name]
	if !ok {
		r.c.mu.Unlock()
		r.Err = srpcErrBadRPCNumber
		r.Done <- r
		return
	}
	r.c.pending[r.id] = r
	if !r.c.muxer {
		r.c.muxer = true
		r.Done <- &yourTurn
	}
	r.c.mu.Unlock()

	var m msg
	m.id = r.id
	m.isRequest = 1
	m.rpc = srv.num
	m.value = arg

	// Fill in the return values and sizes to generate
	// the right type chars.  We'll take most any size.

	// Skip over input arguments.
	// We could check them against arg, but the server
	// will do that anyway.
	i := 0
	for srv.fmt[i] != ':' {
		i++
	}
	format := srv.fmt[i+1:]

	// Now the return prototypes.
	m.template = make([]interface{}, len(format))
	m.size = make([]int, len(format))
	for i := 0; i < len(format); i++ {
		switch format[i] {
		default:
			println("Native Client SRPC: unexpected service type " + string(format[i]))
			r.Err = srpcErrBadRPCNumber
			r.Done <- r
			return
		case 'b':
			m.template[i] = false
		case 'C':
			m.template[i] = []byte(nil)
			m.size[i] = 1 << 30
		case 'd':
			m.template[i] = float64(0)
		case 'D':
			m.template[i] = []float64(nil)
			m.size[i] = 1 << 30
		case 'h':
			m.template[i] = int(-1)
		case 'i':
			m.template[i] = int32(0)
		case 'I':
			m.template[i] = []int32(nil)
			m.size[i] = 1 << 30
		case 's':
			m.template[i] = ""
			m.size[i] = 1 << 30
		}
	}

	if err := m.pack(); err != nil {
		r.Err = errors.New("Native Client RPC Start " + name + ": preparing request: " + err.Error())
		r.Done <- r
		return
	}

	r.c.outMu.Lock()
	r.c.s.send(&m)
	r.c.outMu.Unlock()
}

// Call is a convenience wrapper that starts the RPC request,
// waits for it to finish, and then returns the results.
// Its implementation is:
//
//	r.Start(name, arg)
//	r.Wait()
//	return r.Ret, r.Errno
//
func (c *srpcClient) Call(name string, arg ...interface{}) (ret []interface{}, err error) {
	r := c.NewRPC(nil)
	r.Start(name, arg)
	r.Wait()
	return r.Ret, r.Err
}

// NewRPC creates a new RPC on the client connection.
func (c *srpcClient) NewRPC(done chan *srpc) *srpc {
	if done == nil {
		done = make(chan *srpc, 1)
	}
	c.mu.Lock()
	id := c.idGen
	c.idGen++
	c.mu.Unlock()
	return &srpc{Done: done, c: c, id: id}
}

// The current protocol number.
// Kind of useless, since there have been backwards-incompatible changes
// to the wire protocol that did not update the protocol number.
// At this point it's really just a sanity check.
const protocol = 0xc0da0002

// An srpcErrno is an SRPC status code.
type srpcErrno uint32

const (
	srpcOK srpcErrno = 256 + iota
	srpcErrBreak
	srpcErrMessageTruncated
	srpcErrNoMemory
	srpcErrProtocolMismatch
	srpcErrBadRPCNumber
	srpcErrBadArgType
	srpcErrTooFewArgs
	srpcErrTooManyArgs
	srpcErrInArgTypeMismatch
	srpcErrOutArgTypeMismatch
	srpcErrInternalError
	srpcErrAppError
)

var srpcErrstr = [...]string{
	srpcOK - srpcOK:                    "ok",
	srpcErrBreak - srpcOK:              "break",
	srpcErrMessageTruncated - srpcOK:   "message truncated",
	srpcErrNoMemory - srpcOK:           "out of memory",
	srpcErrProtocolMismatch - srpcOK:   "protocol mismatch",
	srpcErrBadRPCNumber - srpcOK:       "invalid RPC method number",
	srpcErrBadArgType - srpcOK:         "unexpected argument type",
	srpcErrTooFewArgs - srpcOK:         "too few arguments",
	srpcErrTooManyArgs - srpcOK:        "too many arguments",
	srpcErrInArgTypeMismatch - srpcOK:  "input argument type mismatch",
	srpcErrOutArgTypeMismatch - srpcOK: "output argument type mismatch",
	srpcErrInternalError - srpcOK:      "internal error",
	srpcErrAppError - srpcOK:           "application error",
}

func (e srpcErrno) Error() string {
	if e < srpcOK || int(e-srpcOK) >= len(srpcErrstr) {
		return "srpcErrno(" + itoa(int(e)) + ")"
	}
	return srpcErrstr[e-srpcOK]
}

// A msgHdr is the data argument to the imc_recvmsg
// and imc_sendmsg system calls.
type msgHdr struct {
	iov   *iov
	niov  int32
	desc  *int32
	ndesc int32
	flags uint32
}

// A single region for I/O.
type iov struct {
	base *byte
	len  int32
}

const maxMsgSize = 1<<16 - 4*4

// A msgReceiver receives messages from a file descriptor.
type msgReceiver struct {
	fd   int
	data [maxMsgSize]byte
	desc [8]int32
	hdr  msgHdr
	iov  iov
}

func (r *msgReceiver) recv() (*msg, error) {
	// Init pointers to buffers where syscall recvmsg can write.
	r.iov.base = &r.data[0]
	r.iov.len = int32(len(r.data))
	r.hdr.iov = &r.iov
	r.hdr.niov = 1
	r.hdr.desc = &r.desc[0]
	r.hdr.ndesc = int32(len(r.desc))
	n, _, e := Syscall(sys_imc_recvmsg, uintptr(r.fd), uintptr(unsafe.Pointer(&r.hdr)), 0)
	if e != 0 {
		println("Native Client imc_recvmsg: ", e.Error())
		return nil, e
	}

	// Make a copy of the data so that the next recvmsg doesn't
	// smash it.  The system call did not update r.iov.len.  Instead it
	// returned the total byte count as n.
	m := new(msg)
	m.data = make([]byte, n)
	copy(m.data, r.data[0:])

	// Make a copy of the desc too.
	// The system call *did* update r.hdr.ndesc.
	if r.hdr.ndesc > 0 {
		m.desc = make([]int32, r.hdr.ndesc)
		copy(m.desc, r.desc[:])
	}

	return m, nil
}

// A msgSender sends messages on a file descriptor.
type msgSender struct {
	fd  int
	hdr msgHdr
	iov iov
}

func (s *msgSender) send(m *msg) error {
	if len(m.data) > 0 {
		s.iov.base = &m.data[0]
	}
	s.iov.len = int32(len(m.data))
	s.hdr.iov = &s.iov
	s.hdr.niov = 1
	s.hdr.desc = nil
	s.hdr.ndesc = 0
	_, _, e := Syscall(sys_imc_sendmsg, uintptr(s.fd), uintptr(unsafe.Pointer(&s.hdr)), 0)
	if e != 0 {
		println("Native Client imc_sendmsg: ", e.Error())
		return e
	}
	return nil
}

// A msg is the Go representation of an SRPC message.
type msg struct {
	data []byte  // message data
	desc []int32 // message file descriptors

	// parsed version of message
	id        uint32
	isRequest uint32
	rpc       uint32
	status    uint32
	value     []interface{}
	template  []interface{}
	size      []int
	format    string
	broken    bool
}

// reading from a msg

func (m *msg) uint32() uint32 {
	if m.broken {
		return 0
	}
	if len(m.data) < 4 {
		m.broken = true
		return 0
	}
	b := m.data[:4]
	x := uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
	m.data = m.data[4:]
	return x
}

func (m *msg) uint64() uint64 {
	x := uint64(m.uint32()) | uint64(m.uint32())<<32
	if m.broken {
		return 0
	}
	return x
}

func (m *msg) bytes(n int) []byte {
	if m.broken {
		return nil
	}
	if len(m.data) < n {
		m.broken = true
		return nil
	}
	x := m.data[0:n]
	m.data = m.data[n:]
	return x
}

// writing to a msg

func (m *msg) wuint32(x uint32) {
	m.data = append(m.data, byte(x), byte(x>>8), byte(x>>16), byte(x>>24))
}

func (m *msg) wuint64(x uint64) {
	lo := uint32(x)
	hi := uint32(x >> 32)
	m.data = append(m.data, byte(lo), byte(lo>>8), byte(lo>>16), byte(lo>>24), byte(hi), byte(hi>>8), byte(hi>>16), byte(hi>>24))
}

func (m *msg) wbytes(p []byte) {
	m.data = append(m.data, p...)
}

func (m *msg) wstring(s string) {
	m.data = append(m.data, s...)
}

// Parsing of RPC messages.
//
// Each message begins with
//	total_size uint32
//	total_descs uint32
//	fragment_size uint32
//	fragment_descs uint32
//
// If fragment_size < total_size or fragment_descs < total_descs, the actual
// message is broken up in multiple messages; follow-up messages omit
// the "total" fields and begin with the "fragment" fields.
// We do not support putting fragmented messages back together.
// To do this we would need to change the message receiver.
//
// After that size information, the message header follows:
//	protocol uint32
//	requestID uint32
//	isRequest uint32
//	rpcNumber uint32
//	status uint32
//	numValue uint32
//	numTemplate uint32
//
// After the header come numTemplate fixed-size arguments,
// numValue fixed-size arguments, and then the variable-sized
// part of the values. The templates describe the expected results
// and have no associated variable sized data in the request.
//
// Each fixed-size argument has the form:
//	tag uint32 // really a char, like 'b' or 'C'
//	pad uint32 // unused
//	val1 uint32
//	val2 uint32
//
// The tags are:
//	'b':	bool; val1 == 0 or 1
//	'C':	[]byte; val1 == len, data in variable-sized section
//	'd':	float64; (val1, val2) is data
//	'D':	[]float64; val1 == len, data in variable-sized section
//	'h':	int; val1 == file descriptor
//	'i':	int32; descriptor in next entry in m.desc
//	'I':	[]int; val1 == len, data in variable-sized section
//	's':	string; val1 == len, data in variable-sized section
//

func (m *msg) pack() error {
	m.data = m.data[:0]
	m.desc = m.desc[:0]

	// sizes, to fill in later
	m.wuint32(0)
	m.wuint32(0)
	m.wuint32(0)
	m.wuint32(0)

	// message header
	m.wuint32(protocol)
	m.wuint32(m.id)
	m.wuint32(m.isRequest)
	m.wuint32(m.rpc)
	m.wuint32(m.status)
	m.wuint32(uint32(len(m.value)))
	m.wuint32(uint32(len(m.template)))

	// fixed-size templates
	for i, x := range m.template {
		var tag, val1, val2 uint32
		switch x.(type) {
		default:
			return errors.New("unexpected template type")
		case bool:
			tag = 'b'
		case []byte:
			tag = 'C'
			val1 = uint32(m.size[i])
		case float64:
			tag = 'd'
		case []float64:
			tag = 'D'
			val1 = uint32(m.size[i])
		case int:
			tag = 'h'
		case int32:
			tag = 'i'
		case []int32:
			tag = 'I'
			val1 = uint32(m.size[i])
		case string:
			tag = 's'
			val1 = uint32(m.size[i])
		}
		m.wuint32(tag)
		m.wuint32(0)
		m.wuint32(val1)
		m.wuint32(val2)
	}

	// fixed-size values
	for _, x := range m.value {
		var tag, val1, val2 uint32
		switch x := x.(type) {
		default:
			return errors.New("unexpected value type")
		case bool:
			tag = 'b'
			if x {
				val1 = 1
			}
		case []byte:
			tag = 'C'
			val1 = uint32(len(x))
		case float64:
			tag = 'd'
			v := float64bits(x)
			val1 = uint32(v)
			val2 = uint32(v >> 32)
		case []float64:
			tag = 'D'
			val1 = uint32(len(x))
		case int32:
			tag = 'i'
			m.desc = append(m.desc, x)
		case []int32:
			tag = 'I'
			val1 = uint32(len(x))
		case string:
			tag = 's'
			val1 = uint32(len(x) + 1)
		}
		m.wuint32(tag)
		m.wuint32(0)
		m.wuint32(val1)
		m.wuint32(val2)
	}

	// variable-length data for values
	for _, x := range m.value {
		switch x := x.(type) {
		case []byte:
			m.wbytes(x)
		case []float64:
			for _, f := range x {
				m.wuint64(float64bits(f))
			}
		case []int32:
			for _, j := range x {
				m.wuint32(uint32(j))
			}
		case string:
			m.wstring(x)
			m.wstring("\x00")
		}
	}

	// fill in sizes
	data := m.data
	m.data = m.data[:0]
	m.wuint32(uint32(len(data)))
	m.wuint32(uint32(len(m.desc)))
	m.wuint32(uint32(len(data)))
	m.wuint32(uint32(len(m.desc)))
	m.data = data

	return nil
}

func (m *msg) unpack() error {
	totalSize := m.uint32()
	totalDesc := m.uint32()
	fragSize := m.uint32()
	fragDesc := m.uint32()
	if totalSize != fragSize || totalDesc != fragDesc {
		return errors.New("Native Client: fragmented RPC messages not supported")
	}
	if m.uint32() != protocol {
		return errors.New("Native Client: RPC protocol mismatch")
	}

	// message header
	m.id = m.uint32()
	m.isRequest = m.uint32()
	m.rpc = m.uint32()
	m.status = m.uint32()
	m.value = make([]interface{}, m.uint32())
	m.template = make([]interface{}, m.uint32())
	m.size = make([]int, len(m.template))
	if m.broken {
		return errors.New("Native Client: malformed message")
	}

	// fixed-size templates
	for i := range m.template {
		tag := m.uint32()
		m.uint32() // padding
		val1 := m.uint32()
		m.uint32() // val2
		switch tag {
		default:
			return errors.New("Native Client: unexpected template type " + string(rune(tag)))
		case 'b':
			m.template[i] = false
		case 'C':
			m.template[i] = []byte(nil)
			m.size[i] = int(val1)
		case 'd':
			m.template[i] = float64(0)
		case 'D':
			m.template[i] = []float64(nil)
			m.size[i] = int(val1)
		case 'i':
			m.template[i] = int32(0)
		case 'I':
			m.template[i] = []int32(nil)
			m.size[i] = int(val1)
		case 'h':
			m.template[i] = int(0)
		case 's':
			m.template[i] = ""
			m.size[i] = int(val1)
		}
	}

	// fixed-size values
	var (
		strsize []uint32
		d       int
	)
	for i := range m.value {
		tag := m.uint32()
		m.uint32() // padding
		val1 := m.uint32()
		val2 := m.uint32()
		switch tag {
		default:
			return errors.New("Native Client: unexpected value type " + string(rune(tag)))
		case 'b':
			m.value[i] = val1 > 0
		case 'C':
			m.value[i] = []byte(nil)
			strsize = append(strsize, val1)
		case 'd':
			m.value[i] = float64frombits(uint64(val1) | uint64(val2)<<32)
		case 'D':
			m.value[i] = make([]float64, val1)
		case 'i':
			m.value[i] = int32(val1)
		case 'I':
			m.value[i] = make([]int32, val1)
		case 'h':
			m.value[i] = int(m.desc[d])
			d++
		case 's':
			m.value[i] = ""
			strsize = append(strsize, val1)
		}
	}

	// variable-sized parts of values
	for i, x := range m.value {
		switch x := x.(type) {
		case []byte:
			m.value[i] = m.bytes(int(strsize[0]))
			strsize = strsize[1:]
		case []float64:
			for i := range x {
				x[i] = float64frombits(m.uint64())
			}
		case []int32:
			for i := range x {
				x[i] = int32(m.uint32())
			}
		case string:
			m.value[i] = string(m.bytes(int(strsize[0])))
			strsize = strsize[1:]
		}
	}

	if len(m.data) > 0 {
		return errors.New("Native Client: junk at end of message")
	}
	return nil
}

func float64bits(x float64) uint64 {
	return *(*uint64)(unsafe.Pointer(&x))
}

func float64frombits(x uint64) float64 {
	return *(*float64)(unsafe.Pointer(&x))
}

// At startup, connect to the name service.
var nsClient = nsConnect()

func nsConnect() *srpcClient {
	var ns int32 = -1
	_, _, errno := Syscall(sys_nameservice, uintptr(unsafe.Pointer(&ns)), 0, 0)
	if errno != 0 {
		println("Native Client nameservice:", errno.Error())
		return nil
	}

	sock, _, errno := Syscall(sys_imc_connect, uintptr(ns), 0, 0)
	if errno != 0 {
		println("Native Client nameservice connect:", errno.Error())
		return nil
	}

	c, err := newClient(int(sock))
	if err != nil {
		println("Native Client nameservice init:", err.Error())
		return nil
	}

	return c
}

const (
	nsSuccess               = 0
	nsNameNotFound          = 1
	nsDuplicateName         = 2
	nsInsufficientResources = 3
	nsPermissionDenied      = 4
	nsInvalidArgument       = 5
)

func openNamedService(name string, mode int32) (fd int, err error) {
	if nsClient == nil {
		return 0, errors.New("no name service")
	}
	ret, err := nsClient.Call("lookup:si:ih", name, int32(mode))
	if err != nil {
		return 0, err
	}
	status := ret[0].(int32)
	fd = ret[1].(int)
	switch status {
	case nsSuccess:
		// ok
	case nsNameNotFound:
		return -1, ENOENT
	case nsDuplicateName:
		return -1, EEXIST
	case nsInsufficientResources:
		return -1, EWOULDBLOCK
	case nsPermissionDenied:
		return -1, EPERM
	case nsInvalidArgument:
		return -1, EINVAL
	default:
		return -1, EINVAL
	}
	return fd, nil
}
