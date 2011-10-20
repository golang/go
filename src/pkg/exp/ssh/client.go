// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

import (
	"big"
	"crypto"
	"crypto/rand"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"net"
	"sync"
)

// clientVersion is the fixed identification string that the client will use.
var clientVersion = []byte("SSH-2.0-Go\r\n")

// ClientConn represents the client side of an SSH connection.
type ClientConn struct {
	*transport
	config *ClientConfig
	chanlist
}

// Client returns a new SSH client connection using c as the underlying transport.
func Client(c net.Conn, config *ClientConfig) (*ClientConn, os.Error) {
	conn := &ClientConn{
		transport: newTransport(c, config.rand()),
		config:    config,
		chanlist: chanlist{
			Mutex: new(sync.Mutex),
			chans: make(map[uint32]*ClientChan),
		},
	}
	if err := conn.handshake(); err != nil {
		conn.Close()
		return nil, err
	}
	if err := conn.authenticate(); err != nil {
		conn.Close()
		return nil, err
	}
	go conn.mainLoop()
	return conn, nil
}

// handshake performs the client side key exchange. See RFC 4253 Section 7.
func (c *ClientConn) handshake() os.Error {
	var magics handshakeMagics

	if _, err := c.Write(clientVersion); err != nil {
		return err
	}
	if err := c.Flush(); err != nil {
		return err
	}
	magics.clientVersion = clientVersion[:len(clientVersion)-2]

	// read remote server version
	version, err := readVersion(c)
	if err != nil {
		return err
	}
	magics.serverVersion = version
	clientKexInit := kexInitMsg{
		KexAlgos:                supportedKexAlgos,
		ServerHostKeyAlgos:      supportedHostKeyAlgos,
		CiphersClientServer:     supportedCiphers,
		CiphersServerClient:     supportedCiphers,
		MACsClientServer:        supportedMACs,
		MACsServerClient:        supportedMACs,
		CompressionClientServer: supportedCompressions,
		CompressionServerClient: supportedCompressions,
	}
	kexInitPacket := marshal(msgKexInit, clientKexInit)
	magics.clientKexInit = kexInitPacket

	if err := c.writePacket(kexInitPacket); err != nil {
		return err
	}
	packet, err := c.readPacket()
	if err != nil {
		return err
	}

	magics.serverKexInit = packet

	var serverKexInit kexInitMsg
	if err = unmarshal(&serverKexInit, packet, msgKexInit); err != nil {
		return err
	}

	kexAlgo, hostKeyAlgo, ok := findAgreedAlgorithms(c.transport, &clientKexInit, &serverKexInit)
	if !ok {
		return os.NewError("ssh: no common algorithms")
	}

	if serverKexInit.FirstKexFollows && kexAlgo != serverKexInit.KexAlgos[0] {
		// The server sent a Kex message for the wrong algorithm,
		// which we have to ignore.
		if _, err := c.readPacket(); err != nil {
			return err
		}
	}

	var H, K []byte
	var hashFunc crypto.Hash
	switch kexAlgo {
	case kexAlgoDH14SHA1:
		hashFunc = crypto.SHA1
		dhGroup14Once.Do(initDHGroup14)
		H, K, err = c.kexDH(dhGroup14, hashFunc, &magics, hostKeyAlgo)
	default:
		fmt.Errorf("ssh: unexpected key exchange algorithm %v", kexAlgo)
	}
	if err != nil {
		return err
	}

	if err = c.writePacket([]byte{msgNewKeys}); err != nil {
		return err
	}
	if err = c.transport.writer.setupKeys(clientKeys, K, H, H, hashFunc); err != nil {
		return err
	}
	if packet, err = c.readPacket(); err != nil {
		return err
	}
	if packet[0] != msgNewKeys {
		return UnexpectedMessageError{msgNewKeys, packet[0]}
	}
	return c.transport.reader.setupKeys(serverKeys, K, H, H, hashFunc)
}

// authenticate authenticates with the remote server. See RFC 4252. 
// Only "password" authentication is supported.
func (c *ClientConn) authenticate() os.Error {
	if err := c.writePacket(marshal(msgServiceRequest, serviceRequestMsg{serviceUserAuth})); err != nil {
		return err
	}
	packet, err := c.readPacket()
	if err != nil {
		return err
	}

	var serviceAccept serviceAcceptMsg
	if err = unmarshal(&serviceAccept, packet, msgServiceAccept); err != nil {
		return err
	}

	// TODO(dfc) support proper authentication method negotation
	method := "none"
	if c.config.Password != "" {
		method = "password"
	}
	if err := c.sendUserAuthReq(method); err != nil {
		return err
	}

	if packet, err = c.readPacket(); err != nil {
		return err
	}

	if packet[0] != msgUserAuthSuccess {
		return UnexpectedMessageError{msgUserAuthSuccess, packet[0]}
	}
	return nil
}

func (c *ClientConn) sendUserAuthReq(method string) os.Error {
	length := stringLength([]byte(c.config.Password)) + 1
	payload := make([]byte, length)
	// always false for password auth, see RFC 4252 Section 8.
	payload[0] = 0
	marshalString(payload[1:], []byte(c.config.Password))

	return c.writePacket(marshal(msgUserAuthRequest, userAuthRequestMsg{
		User:    c.config.User,
		Service: serviceSSH,
		Method:  method,
		Payload: payload,
	}))
}

// kexDH performs Diffie-Hellman key agreement on a ClientConn. The
// returned values are given the same names as in RFC 4253, section 8.
func (c *ClientConn) kexDH(group *dhGroup, hashFunc crypto.Hash, magics *handshakeMagics, hostKeyAlgo string) ([]byte, []byte, os.Error) {
	x, err := rand.Int(c.config.rand(), group.p)
	if err != nil {
		return nil, nil, err
	}
	X := new(big.Int).Exp(group.g, x, group.p)
	kexDHInit := kexDHInitMsg{
		X: X,
	}
	if err := c.writePacket(marshal(msgKexDHInit, kexDHInit)); err != nil {
		return nil, nil, err
	}

	packet, err := c.readPacket()
	if err != nil {
		return nil, nil, err
	}

	var kexDHReply = new(kexDHReplyMsg)
	if err = unmarshal(kexDHReply, packet, msgKexDHReply); err != nil {
		return nil, nil, err
	}

	if kexDHReply.Y.Sign() == 0 || kexDHReply.Y.Cmp(group.p) >= 0 {
		return nil, nil, os.NewError("server DH parameter out of bounds")
	}

	kInt := new(big.Int).Exp(kexDHReply.Y, x, group.p)
	h := hashFunc.New()
	writeString(h, magics.clientVersion)
	writeString(h, magics.serverVersion)
	writeString(h, magics.clientKexInit)
	writeString(h, magics.serverKexInit)
	writeString(h, kexDHReply.HostKey)
	writeInt(h, X)
	writeInt(h, kexDHReply.Y)
	K := make([]byte, intLength(kInt))
	marshalInt(K, kInt)
	h.Write(K)

	H := h.Sum()

	return H, K, nil
}

// OpenChan opens a new client channel. The most common session type is "session". 
// The full set of valid session types are listed in RFC 4250 4.9.1.
func (c *ClientConn) OpenChan(typ string) (*ClientChan, os.Error) {
	ch, id := c.newChan(c.transport)
	if err := c.writePacket(marshal(msgChannelOpen, channelOpenMsg{
		ChanType:      typ,
		PeersId:       id,
		PeersWindow:   8192,
		MaxPacketSize: 16384,
	})); err != nil {
		// remove channel reference
		c.chanlist.remove(id)
		return nil, err
	}
	// wait for response
	switch msg := (<-ch.msg).(type) {
	case *channelOpenConfirmMsg:
		ch.peersId = msg.MyId
	case *channelOpenFailureMsg:
		c.chanlist.remove(id)
		return nil, os.NewError(msg.Message)
	default:
		c.chanlist.remove(id)
		return nil, os.NewError("Unexpected packet")
	}
	return ch, nil
}

// mainloop reads incoming messages and routes channel messages
// to their respective ClientChans.
func (c *ClientConn) mainLoop() {
	for {
		packet, err := c.readPacket()
		if err != nil {
			// TODO(dfc) signal the underlying close to all channels
			c.Close()
			return
		}
		switch msg := decode(packet).(type) {
		case *channelOpenMsg:
			c.getChan(msg.PeersId).msg <- msg
		case *channelOpenConfirmMsg:
			c.getChan(msg.PeersId).msg <- msg
		case *channelOpenFailureMsg:
			c.getChan(msg.PeersId).msg <- msg
		case *channelCloseMsg:
			ch := c.getChan(msg.PeersId)
			close(ch.stdinWriter.win)
			close(ch.stdoutReader.data)
			close(ch.stderrReader.dataExt)
			c.chanlist.remove(msg.PeersId)
		case *channelEOFMsg:
			c.getChan(msg.PeersId).msg <- msg
		case *channelRequestSuccessMsg:
			c.getChan(msg.PeersId).msg <- msg
		case *channelRequestFailureMsg:
			c.getChan(msg.PeersId).msg <- msg
		case *channelRequestMsg:
			c.getChan(msg.PeersId).msg <- msg
		case *windowAdjustMsg:
			c.getChan(msg.PeersId).stdinWriter.win <- int(msg.AdditionalBytes)
		case *channelData:
			c.getChan(msg.PeersId).stdoutReader.data <- msg.Payload
		case *channelExtendedData:
			// TODO(dfc) should this send be non blocking. RFC 4254 5.2 suggests
			// ext data consumes window size, does that need to be handled as well ?
			c.getChan(msg.PeersId).stderrReader.dataExt <- msg.Data
		default:
			fmt.Printf("mainLoop: unhandled %#v\n", msg)
		}
	}
}

// Dial connects to the given network address using net.Dial and 
// then initiates a SSH handshake, returning the resulting client connection.
func Dial(network, addr string, config *ClientConfig) (*ClientConn, os.Error) {
	conn, err := net.Dial(network, addr)
	if err != nil {
		return nil, err
	}
	return Client(conn, config)
}

// A ClientConfig structure is used to configure a ClientConn. After one has 
// been passed to an SSH function it must not be modified.
type ClientConfig struct {
	// Rand provides the source of entropy for key exchange. If Rand is 
	// nil, the cryptographic random reader in package crypto/rand will 
	// be used.
	Rand io.Reader

	// The username to authenticate.
	User string

	// Used for "password" method authentication.
	Password string
}

func (c *ClientConfig) rand() io.Reader {
	if c.Rand == nil {
		return rand.Reader
	}
	return c.Rand
}

// A ClientChan represents a single RFC 4254 channel that is multiplexed 
// over a single SSH connection.
type ClientChan struct {
	packetWriter
	*stdinWriter  // used by Exec and Shell
	*stdoutReader // used by Exec and Shell
	*stderrReader // used by Exec and Shell
	id, peersId   uint32
	msg           chan interface{} // incoming messages 
}

func newClientChan(t *transport, id uint32) *ClientChan {
	// TODO(DFC) allocating stdin/out/err on ClientChan creation is
	// wasteful, but ClientConn.mainLoop() needs a way of finding 
	// those channels before Exec/Shell is called because the remote 
	// may send window adjustments at any time.
	return &ClientChan{
		packetWriter: t,
		stdinWriter: &stdinWriter{
			packetWriter: t,
			id:           id,
			win:          make(chan int, 16),
		},
		stdoutReader: &stdoutReader{
			packetWriter: t,
			id:           id,
			win:          8192,
			data:         make(chan []byte, 16),
		},
		stderrReader: &stderrReader{
			dataExt: make(chan string, 16),
		},
		id:  id,
		msg: make(chan interface{}, 16),
	}
}

// Close closes the channel. This does not close the underlying connection.
func (c *ClientChan) Close() os.Error {
	return c.writePacket(marshal(msgChannelClose, channelCloseMsg{
		PeersId: c.id,
	}))
}

// Setenv sets an environment variable that will be applied to any
// command executed by Shell or Exec.
func (c *ClientChan) Setenv(name, value string) os.Error {
	namLen := stringLength([]byte(name))
	valLen := stringLength([]byte(value))
	payload := make([]byte, namLen+valLen)
	marshalString(payload[:namLen], []byte(name))
	marshalString(payload[namLen:], []byte(value))

	return c.sendChanReq(channelRequestMsg{
		PeersId:             c.id,
		Request:             "env",
		WantReply:           true,
		RequestSpecificData: payload,
	})
}

func (c *ClientChan) sendChanReq(req channelRequestMsg) os.Error {
	if err := c.writePacket(marshal(msgChannelRequest, req)); err != nil {
		return err
	}
	for {
		switch msg := (<-c.msg).(type) {
		case *channelRequestSuccessMsg:
			return nil
		case *channelRequestFailureMsg:
			return os.NewError(req.Request)
		default:
			return fmt.Errorf("%#v", msg)
		}
	}
	panic("unreachable")
}

// An empty mode list (a string of 1 character, opcode 0), see RFC 4254 Section 8.
var emptyModeList = []byte{0, 0, 0, 1, 0}

// RequstPty requests a pty to be allocated on the remote side of this channel.
func (c *ClientChan) RequestPty(term string, h, w int) os.Error {
	buf := make([]byte, 4+len(term)+16+len(emptyModeList))
	b := marshalString(buf, []byte(term))
	binary.BigEndian.PutUint32(b, uint32(h))
	binary.BigEndian.PutUint32(b[4:], uint32(w))
	binary.BigEndian.PutUint32(b[8:], uint32(h*8))
	binary.BigEndian.PutUint32(b[12:], uint32(w*8))
	copy(b[16:], emptyModeList)

	return c.sendChanReq(channelRequestMsg{
		PeersId:             c.id,
		Request:             "pty-req",
		WantReply:           true,
		RequestSpecificData: buf,
	})
}

// Exec runs cmd on the remote host.
// Typically, the remote server passes cmd to the shell for interpretation.
func (c *ClientChan) Exec(cmd string) (*Cmd, os.Error) {
	cmdLen := stringLength([]byte(cmd))
	payload := make([]byte, cmdLen)
	marshalString(payload, []byte(cmd))
	err := c.sendChanReq(channelRequestMsg{
		PeersId:             c.id,
		Request:             "exec",
		WantReply:           true,
		RequestSpecificData: payload,
	})
	return &Cmd{
		c.stdinWriter,
		c.stdoutReader,
		c.stderrReader,
	}, err
}

// Shell starts a login shell on the remote host.
func (c *ClientChan) Shell() (*Cmd, os.Error) {
	err := c.sendChanReq(channelRequestMsg{
		PeersId:   c.id,
		Request:   "shell",
		WantReply: true,
	})
	return &Cmd{
		c.stdinWriter,
		c.stdoutReader,
		c.stderrReader,
	}, err

}

// Thread safe channel list.
type chanlist struct {
	*sync.Mutex
	// TODO(dfc) should could be converted to a slice
	chans map[uint32]*ClientChan
}

// Allocate a new ClientChan with the next avail local id.
func (c *chanlist) newChan(t *transport) (*ClientChan, uint32) {
	c.Lock()
	defer c.Unlock()

	for i := uint32(0); i < 1<<31; i++ {
		if _, ok := c.chans[i]; !ok {
			ch := newClientChan(t, i)
			c.chans[i] = ch
			return ch, uint32(i)
		}
	}
	panic("unable to find free channel")
}

func (c *chanlist) getChan(id uint32) *ClientChan {
	c.Lock()
	defer c.Unlock()
	return c.chans[id]
}

func (c *chanlist) remove(id uint32) {
	c.Lock()
	defer c.Unlock()
	delete(c.chans, id)
}

// A Cmd represents a connection to a remote command or shell
// Closing Cmd.Stdin will be observed by the remote process.
type Cmd struct {
	// Writes to Stdin are made available to the command's standard input.
	// Closing Stdin causes the command to observe an EOF on its standard input.
	Stdin io.WriteCloser

	// Reads from Stdout consume the command's standard output.
	// There is a fixed amount of buffering of the command's standard output.
	// Failing to read from Stdout will eventually cause the command to block
	// when writing to its standard output.  Closing Stdout unblocks any
	// such writes and makes them return errors.
	Stdout io.ReadCloser

	// Reads from Stderr consume the command's standard error.
	// The SSH protocol assumes it can always send standard error;
	// the command will never block writing to its standard error.
	// However, failure to read from Stderr will eventually cause the
	// SSH protocol to jam, so it is important to arrange for reading
	// from Stderr, even if by
	//    go io.Copy(ioutil.Discard, cmd.Stderr)
	Stderr io.Reader
}

// A stdinWriter represents the stdin of a remote process.
type stdinWriter struct {
	win          chan int // receives window adjustments
	id           uint32
	rwin         int // current rwin size
	packetWriter     // for sending channelDataMsg
}

// Write writes data to the remote process's standard input.
func (w *stdinWriter) Write(data []byte) (n int, err os.Error) {
	for {
		if w.rwin == 0 {
			win, ok := <-w.win
			if !ok {
				return 0, os.EOF
			}
			w.rwin += win
			continue
		}
		n = len(data)
		packet := make([]byte, 0, 9+n)
		packet = append(packet, msgChannelData,
			byte(w.id)>>24, byte(w.id)>>16, byte(w.id)>>8, byte(w.id),
			byte(n)>>24, byte(n)>>16, byte(n)>>8, byte(n))
		err = w.writePacket(append(packet, data...))
		w.rwin -= n
		return
	}
	panic("unreachable")
}

func (w *stdinWriter) Close() os.Error {
	return w.writePacket(marshal(msgChannelEOF, channelEOFMsg{w.id}))
}

// A stdoutReader represents the stdout of a remote process.
type stdoutReader struct {
	// TODO(dfc) a fixed size channel may not be the right data structure.
	// If writes to this channel block, they will block mainLoop, making
	// it unable to receive new messages from the remote side.
	data         chan []byte // receives data from remote
	id           uint32
	win          int // current win size
	packetWriter     // for sending windowAdjustMsg
	buf          []byte
}

// Read reads data from the remote process's standard output.
func (r *stdoutReader) Read(data []byte) (int, os.Error) {
	var ok bool
	for {
		if len(r.buf) > 0 {
			n := copy(data, r.buf)
			r.buf = r.buf[n:]
			r.win += n
			msg := windowAdjustMsg{
				PeersId:         r.id,
				AdditionalBytes: uint32(n),
			}
			err := r.writePacket(marshal(msgChannelWindowAdjust, msg))
			return n, err
		}
		r.buf, ok = <-r.data
		if !ok {
			return 0, os.EOF
		}
		r.win -= len(r.buf)
	}
	panic("unreachable")
}

func (r *stdoutReader) Close() os.Error {
	return r.writePacket(marshal(msgChannelEOF, channelEOFMsg{r.id}))
}

// A stderrReader represents the stderr of a remote process.
type stderrReader struct {
	dataExt chan string // receives dataExt from remote
	buf     []byte      // buffer current dataExt
}

// Read reads a line of data from the remote process's stderr.
func (r *stderrReader) Read(data []byte) (int, os.Error) {
	for {
		if len(r.buf) > 0 {
			n := copy(data, r.buf)
			r.buf = r.buf[n:]
			return n, nil
		}
		buf, ok := <-r.dataExt
		if !ok {
			return 0, os.EOF
		}
		r.buf = []byte(buf)
	}
	panic("unreachable")
}
