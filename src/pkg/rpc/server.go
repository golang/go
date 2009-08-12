// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
	The rpc package provides access to the public methods of an object across a
	network or other I/O connection.  A server registers an object, making it visible
	as a service with the name of the type of the object.  After registration, public
	methods of the object will be accessible remotely.  A server may register multiple
	objects (services) of different types but it is an error to register multiple
	objects of the same type.

	Only methods that satisfy these criteria will be made available for remote access;
	other methods will be ignored:

		- the method name is publicly visible, that is, begins with an upper case letter.
		- the method has two arguments, both pointers to publicly visible structs.
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
	call, a structure containing the arguments, and a structure to receive the result
	parameters.

	Call waits for the remote call to complete; Go launches the call asynchronously
	and returns a channel that will signal completion.

	Package "gob" is used to transport the data.

	Here is a simple example.  A server wishes to export an object of type Arith:

		package server

		type Args struct {
			A, B int
		}

		type Reply struct {
			C int
		}

		type Arith int

		func (t *Arith) Multiply(args *Args, reply *Reply) os.Error {
			reply.C = args.A * args.B;
			return nil
		}

		func (t *Arith) Divide(args *Args, reply *Reply) os.Error {
			if args.B == 0 {
				return os.ErrorString("divide by zero");
			}
			reply.C = args.A / args.B;
			return nil
		}

	The server calls (for HTTP service):

		arith := new(Arith);
		rpc.Register(arith);
		rpc.HandleHTTP();
		l, e := net.Listen("tcp", ":1234");
		if e != nil {
			log.Exit("listen error:", e);
		}
		go http.Serve(l, nil);

	At this point, clients can see a service "Arith" with methods "Arith.Multiply" and
	"Arith.Divide".  To invoke one, a client first dials the server:

		client, err := rpc.DialHTTP("tcp", serverAddress + ":1234");
		if err != nil {
			log.Exit("dialing:", err);
		}

	Then it can make a remote call:

		// Synchronous call
		args := &server.Args{7,8};
		reply := new(server.Reply);
		err = client.Call("Arith.Multiply", args, reply);
		if err != nil {
			log.Exit("arith error:", err);
		}
		fmt.Printf("Arith: %d*%d=%d", args.A, args.B, reply.C);

	or

		// Asynchronous call
		divCall := client.Go("Arith.Divide", args, reply, nil);
		replyCall := <-divCall.Done;	// will be equal to divCall
		// check errors, print, etc.

	A server implementation will often provide a simple, type-safe wrapper for the
	client.
*/
package rpc

import (
	"gob";
	"http";
	"log";
	"io";
	"net";
	"os";
	"reflect";
	"strings";
	"sync";
	"unicode";
	"utf8";
)

// Precompute the reflect type for os.Error.  Can't use os.Error directly
// because Typeof takes an empty interface value.  This is annoying.
var unusedError *os.Error;
var typeOfOsError = reflect.Typeof(unusedError).(*reflect.PtrType).Elem()

type methodType struct {
	sync.Mutex;	// protects counters
	method	reflect.Method;
	argType	*reflect.PtrType;
	replyType	*reflect.PtrType;
	numCalls	uint;
}

type service struct {
	name	string;	// name of service
	rcvr	reflect.Value;	// receiver of methods for the service
	typ	reflect.Type;	// type of the receiver
	method	map[string] *methodType;	// registered methods
}

// Request is a header written before every RPC call.  It is used internally
// but documented here as an aid to debugging, such as when analyzing
// network traffic.
type Request struct {
	ServiceMethod	string;	// format: "Service.Method"
	Seq	uint64;	// sequence number chosen by client
}

// Response is a header written before every RPC return.  It is used internally
// but documented here as an aid to debugging, such as when analyzing
// network traffic.
type Response struct {
	ServiceMethod	string;	// echoes that of the Request
	Seq	uint64;	// echoes that of the request
	Error	string;	// error, if any.
}

type serverType struct {
	sync.Mutex;	// protects the serviceMap
	serviceMap	map[string] *service;
}

// This variable is a global whose "public" methods are really private methods
// called from the global functions of this package: rpc.Register, rpc.ServeConn, etc.
// For example, rpc.Register() calls server.add().
var server = &serverType{ serviceMap: make(map[string] *service) }

// Is this a publicly vislble - upper case - name?
func isPublic(name string) bool {
	rune, wid_ := utf8.DecodeRuneInString(name);
	return unicode.IsUpper(rune)
}

func (server *serverType) register(rcvr interface{}) os.Error {
	server.Lock();
	defer server.Unlock();
	if server.serviceMap == nil {
		server.serviceMap = make(map[string] *service);
	}
	s := new(service);
	s.typ = reflect.Typeof(rcvr);
	s.rcvr = reflect.NewValue(rcvr);
	sname := reflect.Indirect(s.rcvr).Type().Name();
	if sname == "" {
		log.Exit("rpc: no service name for type", s.typ.String())
	}
	if !isPublic(sname) {
		s := "rpc Register: type " + sname + " is not public";
		log.Stderr(s);
		return os.ErrorString(s);
	}
	if _, present := server.serviceMap[sname]; present {
		return os.ErrorString("rpc: service already defined: " + sname);
	}
	s.name = sname;
	s.method = make(map[string] *methodType);

	// Install the methods
	for m := 0; m < s.typ.NumMethod(); m++ {
		method := s.typ.Method(m);
		mtype := method.Type;
		mname := method.Name;
		if !isPublic(mname) {
			continue
		}
		// Method needs three ins: receiver, *args, *reply.
		// The args and reply must be structs until gobs are more general.
		if mtype.NumIn() != 3 {
			log.Stderr("method", mname, "has wrong number of ins:", mtype.NumIn());
			continue;
		}
		argType, ok := mtype.In(1).(*reflect.PtrType);
		if !ok {
			log.Stderr(mname, "arg type not a pointer:", argType.String());
			continue;
		}
		if _, ok := argType.Elem().(*reflect.StructType); !ok {
			log.Stderr(mname, "arg type not a pointer to a struct:", argType.String());
			continue;
		}
		replyType, ok := mtype.In(2).(*reflect.PtrType);
		if !ok {
			log.Stderr(mname, "reply type not a pointer:", replyType.String());
			continue;
		}
		if _, ok := replyType.Elem().(*reflect.StructType); !ok {
			log.Stderr(mname, "reply type not a pointer to a struct:", replyType.String());
			continue;
		}
		if !isPublic(argType.Elem().Name()) {
			log.Stderr(mname, "argument type not public:", argType.String());
			continue;
		}
		if !isPublic(replyType.Elem().Name()) {
			log.Stderr(mname, "reply type not public:", replyType.String());
			continue;
		}
		// Method needs one out: os.Error.
		if mtype.NumOut() != 1 {
			log.Stderr("method", mname, "has wrong number of outs:", mtype.NumOut());
			continue;
		}
		if returnType := mtype.Out(0); returnType != typeOfOsError {
			log.Stderr("method", mname, "returns", returnType.String(), "not os.Error");
			continue;
		}
		s.method[mname] = &methodType{method: method, argType: argType, replyType: replyType};
	}

	if len(s.method) == 0 {
		s := "rpc Register: type " + sname + " has no public methods of suitable type";
		log.Stderr(s);
		return os.ErrorString(s);
	}
	server.serviceMap[s.name] = s;
	return nil;
}

// A value sent as a placeholder for the response when the server receives an invalid request.
type InvalidRequest struct {
	marker int
}
var invalidRequest = InvalidRequest{1}

func _new(t *reflect.PtrType) *reflect.PtrValue {
	v := reflect.MakeZero(t).(*reflect.PtrValue);
	v.PointTo(reflect.MakeZero(t.Elem()));
	return v;
}

func sendResponse(sending *sync.Mutex, req *Request, reply interface{}, enc *gob.Encoder, errmsg string) {
	resp := new(Response);
	// Encode the response header
	resp.ServiceMethod = req.ServiceMethod;
	resp.Error = errmsg;
	resp.Seq = req.Seq;
	sending.Lock();
	enc.Encode(resp);
	// Encode the reply value.
	enc.Encode(reply);
	sending.Unlock();
}

func (s *service) call(sending *sync.Mutex, mtype *methodType, req *Request, argv, replyv reflect.Value, enc *gob.Encoder) {
	mtype.Lock();
	mtype.numCalls++;
	mtype.Unlock();
	function := mtype.method.Func;
	// Invoke the method, providing a new value for the reply.
	returnValues := function.Call([]reflect.Value{s.rcvr, argv, replyv});
	// The return value for the method is an os.Error.
	errInter := returnValues[0].Interface();
	errmsg := "";
	if errInter != nil {
		errmsg = errInter.(os.Error).String();
	}
	sendResponse(sending, req, replyv.Interface(), enc, errmsg);
}

func (server *serverType) input(conn io.ReadWriteCloser) {
	dec := gob.NewDecoder(conn);
	enc := gob.NewEncoder(conn);
	sending := new(sync.Mutex);
	for {
		// Grab the request header.
		req := new(Request);
		err := dec.Decode(req);
		if err != nil {
			if err == os.EOF || err == io.ErrUnexpectedEOF {
				log.Stderr("rpc: ", err);
				break;
			}
			s := "rpc: server cannot decode request: " + err.String();
			sendResponse(sending, req, invalidRequest, enc, s);
			continue;
		}
		serviceMethod := strings.Split(req.ServiceMethod, ".", 0);
		if len(serviceMethod) != 2 {
			s := "rpc: service/method request ill:formed: " + req.ServiceMethod;
			sendResponse(sending, req, invalidRequest, enc, s);
			continue;
		}
		// Look up the request.
		server.Lock();
		service, ok := server.serviceMap[serviceMethod[0]];
		server.Unlock();
		if !ok {
			s := "rpc: can't find service " + req.ServiceMethod;
			sendResponse(sending, req, invalidRequest, enc, s);
			continue;
		}
		mtype, ok := service.method[serviceMethod[1]];
		if !ok {
			s := "rpc: can't find method " + req.ServiceMethod;
			sendResponse(sending, req, invalidRequest, enc, s);
			continue;
		}
		method := mtype.method;
		// Decode the argument value.
		argv := _new(mtype.argType);
		replyv := _new(mtype.replyType);
		err = dec.Decode(argv.Interface());
		if err != nil {
			log.Stderr("rpc: tearing down", serviceMethod[0], "connection:", err);
			sendResponse(sending, req, replyv.Interface(), enc, err.String());
			continue;
		}
		go service.call(sending, mtype, req, argv, replyv, enc);
	}
	conn.Close();
}

func (server *serverType) accept(lis net.Listener) {
	for {
		conn, addr, err := lis.Accept();
		if err != nil {
			log.Exit("rpc.Serve: accept:", err.String());	// TODO(r): exit?
		}
		go server.input(conn);
	}
}

// Register publishes in the server the set of methods of the
// receiver value that satisfy the following conditions:
//	- public method
//	- two arguments, both pointers to public structs
//	- one return value of type os.Error
// It returns an error if the receiver is not public or has no
// suitable methods.
func Register(rcvr interface{}) os.Error {
	return server.register(rcvr)
}

// ServeConn runs the server on a single connection.  When the connection
// completes, service terminates.  ServeConn blocks; the caller typically
// invokes it in a go statement.
func ServeConn(conn io.ReadWriteCloser) {
	go server.input(conn)
}

// Accept accepts connections on the listener and serves requests
// for each incoming connection.  Accept blocks; the caller typically
// invokes it in a go statement.
func Accept(lis net.Listener) {
	server.accept(lis)
}

// Can connect to RPC service using HTTP CONNECT to rpcPath.
var rpcPath string = "/_goRPC_"
var debugPath string = "/debug/rpc"
var connected = "200 Connected to Go RPC"

func serveHTTP(c *http.Conn, req *http.Request) {
	if req.Method != "CONNECT" {
		c.SetHeader("Content-Type", "text/plain; charset=utf-8");
		c.WriteHeader(http.StatusMethodNotAllowed);
		io.WriteString(c, "405 must CONNECT to " + rpcPath + "\n");
		return;
	}
	conn, buf, err := c.Hijack();
	if err != nil {
		log.Stderr("rpc hijacking ", c.RemoteAddr, ": ", err.String());
		return;
	}
	io.WriteString(conn, "HTTP/1.0 " + connected + "\n\n");
	server.input(conn);
}

// HandleHTTP registers an HTTP handler for RPC messages.
// It is still necessary to invoke http.Serve(), typically in a go statement.
func HandleHTTP() {
	http.Handle(rpcPath, http.HandlerFunc(serveHTTP));
	http.Handle(debugPath, http.HandlerFunc(debugHTTP));
}
