// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
	The rpc package provides access to the exported methods of an object across a
	network or other I/O connection.  A server registers an object, making it visible
	as a service with the name of the type of the object.  After registration, exported
	methods of the object will be accessible remotely.  A server may register multiple
	objects (services) of different types but it is an error to register multiple
	objects of the same type.

	Only methods that satisfy these criteria will be made available for remote access;
	other methods will be ignored:

		- the method receiver and name are exported, that is, begin with an upper case letter.
		- the method has two arguments, both pointers to exported types.
		- the method has return type os.Error.

	The method's first argument represents the arguments provided by the caller; the
	second argument represents the result parameters to be returned to the caller.
	The method's return value, if non-nil, is passed back as a string that the client
	sees as an os.ErrorString.

	The server may handle requests on a single connection by calling ServeConn.  More
	typically it will create a network listener and call Accept or, for an HTTP
	listener, HandleHTTP and http.Serve.

	A client wishing to use the service establishes a connection and then invokes
	NewClient on the connection.  The convenience function Dial (DialHTTP) performs
	both steps for a raw network connection (an HTTP connection).  The resulting
	Client object has two methods, Call and Go, that specify the service and method to
	call, a pointer containing the arguments, and a pointer to receive the result
	parameters.

	Call waits for the remote call to complete; Go launches the call asynchronously
	and returns a channel that will signal completion.

	Package "gob" is used to transport the data.

	Here is a simple example.  A server wishes to export an object of type Arith:

		package server

		type Args struct {
			A, B int
		}

		type Quotient struct {
			Quo, Rem int
		}

		type Arith int

		func (t *Arith) Multiply(args *Args, reply *int) os.Error {
			*reply = args.A * args.B
			return nil
		}

		func (t *Arith) Divide(args *Args, quo *Quotient) os.Error {
			if args.B == 0 {
				return os.ErrorString("divide by zero")
			}
			quo.Quo = args.A / args.B
			quo.Rem = args.A % args.B
			return nil
		}

	The server calls (for HTTP service):

		arith := new(Arith)
		rpc.Register(arith)
		rpc.HandleHTTP()
		l, e := net.Listen("tcp", ":1234")
		if e != nil {
			log.Exit("listen error:", e)
		}
		go http.Serve(l, nil)

	At this point, clients can see a service "Arith" with methods "Arith.Multiply" and
	"Arith.Divide".  To invoke one, a client first dials the server:

		client, err := rpc.DialHTTP("tcp", serverAddress + ":1234")
		if err != nil {
			log.Exit("dialing:", err)
		}

	Then it can make a remote call:

		// Synchronous call
		args := &server.Args{7,8}
		var reply int
		err = client.Call("Arith.Multiply", args, &reply)
		if err != nil {
			log.Exit("arith error:", err)
		}
		fmt.Printf("Arith: %d*%d=%d", args.A, args.B, *reply)

	or

		// Asynchronous call
		quotient := new(Quotient)
		divCall := client.Go("Arith.Divide", args, &quotient, nil)
		replyCall := <-divCall.Done	// will be equal to divCall
		// check errors, print, etc.

	A server implementation will often provide a simple, type-safe wrapper for the
	client.
*/
package rpc

import (
	"gob"
	"http"
	"log"
	"io"
	"net"
	"os"
	"reflect"
	"strings"
	"sync"
	"unicode"
	"utf8"
)

const (
	// Defaults used by HandleHTTP
	DefaultRPCPath   = "/_goRPC_"
	DefaultDebugPath = "/debug/rpc"
)

// Precompute the reflect type for os.Error.  Can't use os.Error directly
// because Typeof takes an empty interface value.  This is annoying.
var unusedError *os.Error
var typeOfOsError = reflect.Typeof(unusedError).(*reflect.PtrType).Elem()

type methodType struct {
	sync.Mutex // protects counters
	method     reflect.Method
	ArgType    *reflect.PtrType
	ReplyType  *reflect.PtrType
	numCalls   uint
}

type service struct {
	name   string                 // name of service
	rcvr   reflect.Value          // receiver of methods for the service
	typ    reflect.Type           // type of the receiver
	method map[string]*methodType // registered methods
}

// Request is a header written before every RPC call.  It is used internally
// but documented here as an aid to debugging, such as when analyzing
// network traffic.
type Request struct {
	ServiceMethod string // format: "Service.Method"
	Seq           uint64 // sequence number chosen by client
}

// Response is a header written before every RPC return.  It is used internally
// but documented here as an aid to debugging, such as when analyzing
// network traffic.
type Response struct {
	ServiceMethod string // echoes that of the Request
	Seq           uint64 // echoes that of the request
	Error         string // error, if any.
}

// ClientInfo records information about an RPC client connection.
type ClientInfo struct {
	LocalAddr  string
	RemoteAddr string
}

// Server represents an RPC Server.
type Server struct {
	sync.Mutex // protects the serviceMap
	serviceMap map[string]*service
}

// NewServer returns a new Server.
func NewServer() *Server {
	return &Server{serviceMap: make(map[string]*service)}
}

// DefaultServer is the default instance of *Server.
var DefaultServer = NewServer()

// Is this an exported - upper case - name?
func isExported(name string) bool {
	rune, _ := utf8.DecodeRuneInString(name)
	return unicode.IsUpper(rune)
}

// Register publishes in the server the set of methods of the
// receiver value that satisfy the following conditions:
//	- exported method
//	- two arguments, both pointers to exported structs
//	- one return value, of type os.Error
// It returns an error if the receiver is not an exported type or has no
// suitable methods.
// The client accesses each method using a string of the form "Type.Method",
// where Type is the receiver's concrete type.
func (server *Server) Register(rcvr interface{}) os.Error {
	return server.register(rcvr, "", false)
}

// RegisterName is like Register but uses the provided name for the type 
// instead of the receiver's concrete type.
func (server *Server) RegisterName(name string, rcvr interface{}) os.Error {
	return server.register(rcvr, name, true)
}

func (server *Server) register(rcvr interface{}, name string, useName bool) os.Error {
	server.Lock()
	defer server.Unlock()
	if server.serviceMap == nil {
		server.serviceMap = make(map[string]*service)
	}
	s := new(service)
	s.typ = reflect.Typeof(rcvr)
	s.rcvr = reflect.NewValue(rcvr)
	sname := reflect.Indirect(s.rcvr).Type().Name()
	if useName {
		sname = name
	}
	if sname == "" {
		log.Exit("rpc: no service name for type", s.typ.String())
	}
	if s.typ.PkgPath() != "" && !isExported(sname) && !useName {
		s := "rpc Register: type " + sname + " is not exported"
		log.Print(s)
		return os.ErrorString(s)
	}
	if _, present := server.serviceMap[sname]; present {
		return os.ErrorString("rpc: service already defined: " + sname)
	}
	s.name = sname
	s.method = make(map[string]*methodType)

	// Install the methods
	for m := 0; m < s.typ.NumMethod(); m++ {
		method := s.typ.Method(m)
		mtype := method.Type
		mname := method.Name
		if mtype.PkgPath() != "" || !isExported(mname) {
			continue
		}
		// Method needs three ins: receiver, *args, *reply.
		if mtype.NumIn() != 3 {
			log.Println("method", mname, "has wrong number of ins:", mtype.NumIn())
			continue
		}
		argType, ok := mtype.In(1).(*reflect.PtrType)
		if !ok {
			log.Println(mname, "arg type not a pointer:", mtype.In(1))
			continue
		}
		replyType, ok := mtype.In(2).(*reflect.PtrType)
		if !ok {
			log.Println(mname, "reply type not a pointer:", mtype.In(2))
			continue
		}
		if argType.Elem().PkgPath() != "" && !isExported(argType.Elem().Name()) {
			log.Println(mname, "argument type not exported:", argType)
			continue
		}
		if replyType.Elem().PkgPath() != "" && !isExported(replyType.Elem().Name()) {
			log.Println(mname, "reply type not exported:", replyType)
			continue
		}
		if mtype.NumIn() == 4 {
			t := mtype.In(3)
			if t != reflect.Typeof((*ClientInfo)(nil)) {
				log.Println(mname, "last argument not *ClientInfo")
				continue
			}
		}
		// Method needs one out: os.Error.
		if mtype.NumOut() != 1 {
			log.Println("method", mname, "has wrong number of outs:", mtype.NumOut())
			continue
		}
		if returnType := mtype.Out(0); returnType != typeOfOsError {
			log.Println("method", mname, "returns", returnType.String(), "not os.Error")
			continue
		}
		s.method[mname] = &methodType{method: method, ArgType: argType, ReplyType: replyType}
	}

	if len(s.method) == 0 {
		s := "rpc Register: type " + sname + " has no exported methods of suitable type"
		log.Print(s)
		return os.ErrorString(s)
	}
	server.serviceMap[s.name] = s
	return nil
}

// A value sent as a placeholder for the response when the server receives an invalid request.
type InvalidRequest struct {
	marker int
}

var invalidRequest = InvalidRequest{1}

func _new(t *reflect.PtrType) *reflect.PtrValue {
	v := reflect.MakeZero(t).(*reflect.PtrValue)
	v.PointTo(reflect.MakeZero(t.Elem()))
	return v
}

func sendResponse(sending *sync.Mutex, req *Request, reply interface{}, codec ServerCodec, errmsg string) {
	resp := new(Response)
	// Encode the response header
	resp.ServiceMethod = req.ServiceMethod
	if errmsg != "" {
		resp.Error = errmsg
	}
	resp.Seq = req.Seq
	sending.Lock()
	err := codec.WriteResponse(resp, reply)
	if err != nil {
		log.Println("rpc: writing response:", err)
	}
	sending.Unlock()
}

func (m *methodType) NumCalls() (n uint) {
	m.Lock()
	n = m.numCalls
	m.Unlock()
	return n
}

func (s *service) call(sending *sync.Mutex, mtype *methodType, req *Request, argv, replyv reflect.Value, codec ServerCodec) {
	mtype.Lock()
	mtype.numCalls++
	mtype.Unlock()
	function := mtype.method.Func
	// Invoke the method, providing a new value for the reply.
	returnValues := function.Call([]reflect.Value{s.rcvr, argv, replyv})
	// The return value for the method is an os.Error.
	errInter := returnValues[0].Interface()
	errmsg := ""
	if errInter != nil {
		errmsg = errInter.(os.Error).String()
	}
	sendResponse(sending, req, replyv.Interface(), codec, errmsg)
}

type gobServerCodec struct {
	rwc io.ReadWriteCloser
	dec *gob.Decoder
	enc *gob.Encoder
}

func (c *gobServerCodec) ReadRequestHeader(r *Request) os.Error {
	return c.dec.Decode(r)
}

func (c *gobServerCodec) ReadRequestBody(body interface{}) os.Error {
	return c.dec.Decode(body)
}

func (c *gobServerCodec) WriteResponse(r *Response, body interface{}) os.Error {
	if err := c.enc.Encode(r); err != nil {
		return err
	}
	return c.enc.Encode(body)
}

func (c *gobServerCodec) Close() os.Error {
	return c.rwc.Close()
}


// ServeConn runs the server on a single connection.
// ServeConn blocks, serving the connection until the client hangs up.
// The caller typically invokes ServeConn in a go statement.
// ServeConn uses the gob wire format (see package gob) on the
// connection.  To use an alternate codec, use ServeCodec.
func (server *Server) ServeConn(conn io.ReadWriteCloser) {
	server.ServeCodec(&gobServerCodec{conn, gob.NewDecoder(conn), gob.NewEncoder(conn)})
}

// ServeCodec is like ServeConn but uses the specified codec to
// decode requests and encode responses.
func (server *Server) ServeCodec(codec ServerCodec) {
	sending := new(sync.Mutex)
	for {
		// Grab the request header.
		req := new(Request)
		err := codec.ReadRequestHeader(req)
		if err != nil {
			if err == os.EOF || err == io.ErrUnexpectedEOF {
				if err == io.ErrUnexpectedEOF {
					log.Println("rpc:", err)
				}
				break
			}
			s := "rpc: server cannot decode request: " + err.String()
			sendResponse(sending, req, invalidRequest, codec, s)
			break
		}
		serviceMethod := strings.Split(req.ServiceMethod, ".", -1)
		if len(serviceMethod) != 2 {
			s := "rpc: service/method request ill-formed: " + req.ServiceMethod
			sendResponse(sending, req, invalidRequest, codec, s)
			continue
		}
		// Look up the request.
		server.Lock()
		service, ok := server.serviceMap[serviceMethod[0]]
		server.Unlock()
		if !ok {
			s := "rpc: can't find service " + req.ServiceMethod
			sendResponse(sending, req, invalidRequest, codec, s)
			continue
		}
		mtype, ok := service.method[serviceMethod[1]]
		if !ok {
			s := "rpc: can't find method " + req.ServiceMethod
			sendResponse(sending, req, invalidRequest, codec, s)
			continue
		}
		// Decode the argument value.
		argv := _new(mtype.ArgType)
		replyv := _new(mtype.ReplyType)
		err = codec.ReadRequestBody(argv.Interface())
		if err != nil {
			log.Println("rpc: tearing down", serviceMethod[0], "connection:", err)
			sendResponse(sending, req, replyv.Interface(), codec, err.String())
			break
		}
		go service.call(sending, mtype, req, argv, replyv, codec)
	}
	codec.Close()
}

// Accept accepts connections on the listener and serves requests
// for each incoming connection.  Accept blocks; the caller typically
// invokes it in a go statement.
func (server *Server) Accept(lis net.Listener) {
	for {
		conn, err := lis.Accept()
		if err != nil {
			log.Exit("rpc.Serve: accept:", err.String()) // TODO(r): exit?
		}
		go server.ServeConn(conn)
	}
}

// Register publishes the receiver's methods in the DefaultServer.
func Register(rcvr interface{}) os.Error { return DefaultServer.Register(rcvr) }

// RegisterName is like Register but uses the provided name for the type 
// instead of the receiver's concrete type.
func RegisterName(name string, rcvr interface{}) os.Error {
	return DefaultServer.RegisterName(name, rcvr)
}

// A ServerCodec implements reading of RPC requests and writing of
// RPC responses for the server side of an RPC session.
// The server calls ReadRequestHeader and ReadRequestBody in pairs
// to read requests from the connection, and it calls WriteResponse to
// write a response back.  The server calls Close when finished with the
// connection.
type ServerCodec interface {
	ReadRequestHeader(*Request) os.Error
	ReadRequestBody(interface{}) os.Error
	WriteResponse(*Response, interface{}) os.Error

	Close() os.Error
}

// ServeConn runs the DefaultServer on a single connection.
// ServeConn blocks, serving the connection until the client hangs up.
// The caller typically invokes ServeConn in a go statement.
// ServeConn uses the gob wire format (see package gob) on the
// connection.  To use an alternate codec, use ServeCodec.
func ServeConn(conn io.ReadWriteCloser) {
	DefaultServer.ServeConn(conn)
}

// ServeCodec is like ServeConn but uses the specified codec to
// decode requests and encode responses.
func ServeCodec(codec ServerCodec) {
	DefaultServer.ServeCodec(codec)
}

// Accept accepts connections on the listener and serves requests
// to DefaultServer for each incoming connection.  
// Accept blocks; the caller typically invokes it in a go statement.
func Accept(lis net.Listener) { DefaultServer.Accept(lis) }

// Can connect to RPC service using HTTP CONNECT to rpcPath.
var connected = "200 Connected to Go RPC"

// ServeHTTP implements an http.Handler that answers RPC requests.
func (server *Server) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	if req.Method != "CONNECT" {
		w.SetHeader("Content-Type", "text/plain; charset=utf-8")
		w.WriteHeader(http.StatusMethodNotAllowed)
		io.WriteString(w, "405 must CONNECT\n")
		return
	}
	conn, _, err := w.Hijack()
	if err != nil {
		log.Print("rpc hijacking ", w.RemoteAddr(), ": ", err.String())
		return
	}
	io.WriteString(conn, "HTTP/1.0 "+connected+"\n\n")
	server.ServeConn(conn)
}

// HandleHTTP registers an HTTP handler for RPC messages on rpcPath,
// and a debugging handler on debugPath.
// It is still necessary to invoke http.Serve(), typically in a go statement.
func (server *Server) HandleHTTP(rpcPath, debugPath string) {
	http.Handle(rpcPath, server)
	http.Handle(debugPath, debugHTTP{server})
}

// HandleHTTP registers an HTTP handler for RPC messages to DefaultServer
// on DefaultRPCPath and a debugging handler on DefaultDebugPath.
// It is still necessary to invoke http.Serve(), typically in a go statement.
func HandleHTTP() {
	DefaultServer.HandleHTTP(DefaultRPCPath, DefaultDebugPath)
}
