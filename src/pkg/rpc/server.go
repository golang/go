// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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
	method	reflect.Method;
	argType	*reflect.PtrType;
	replyType	*reflect.PtrType;
}

type service struct {
	name	string;	// name of service
	rcvr	reflect.Value;	// receiver of methods for the service
	typ	reflect.Type;	// type of the receiver
	method	map[string] *methodType;	// registered methods
}

// Request is a header written before every RPC call.
type Request struct {
	ServiceMethod	string;
	Seq	uint64;
}

// Response is a header written before every RPC return.
type Response struct {
	ServiceMethod	string;
	Seq	uint64;
	Error	string;
}

type serverType struct {
	serviceMap	map[string] *service;
}

// This variable is a global whose "public" methods are really private methods
// called from the global functions of this package: rpc.Add, rpc.ServeConn, etc.
// For example, rpc.Add() calls server.add().
var server = &serverType{ make(map[string] *service) }

// Is this a publicly vislble - upper case - name?
func isPublic(name string) bool {
	rune, wid_ := utf8.DecodeRuneInString(name);
	return unicode.IsUpper(rune)
}

func (server *serverType) add(rcvr interface{}) os.Error {
	if server.serviceMap == nil {
		server.serviceMap = make(map[string] *service);
	}
	s := new(service);
	s.typ = reflect.Typeof(rcvr);
	s.rcvr = reflect.NewValue(rcvr);
	path_, sname := reflect.Indirect(s.rcvr).Type().Name();
	if sname == "" {
		log.Exit("rpc: no service name for type", s.typ.String())
	}
	if !isPublic(sname) {
		s := "rpc Add: type " + sname + " is not public";
		log.Stderr(s);
		return os.ErrorString(s);
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
		// Method needs one out: os.Error.
		if mtype.NumOut() != 1 {
			log.Stderr("method", mname, "has wrong number of outs:", mtype.NumOut());
			continue;
		}
		if returnType := mtype.Out(0); returnType != typeOfOsError {
			log.Stderr("method", mname, "returns", returnType.String(), "not os.Error");
			continue;
		}
		s.method[mname] = &methodType{method, argType, replyType};
	}

	if len(s.method) == 0 {
		s := "rpc Add: type " + sname + " has no public methods of suitable type";
		log.Stderr(s);
		return os.ErrorString(s);
	}
	server.serviceMap[s.name] = s;
	return nil;
}

// A value to be sent as a placeholder for the response when we receive invalid request.
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

func (s *service) call(sending *sync.Mutex, function *reflect.FuncValue, req *Request, argv, replyv reflect.Value, enc *gob.Encoder) {
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
		service, ok := server.serviceMap[serviceMethod[0]];
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
		go service.call(sending, method.Func, req, argv, replyv, enc);
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

// Add publishes in the server the set of methods of the
// receiver value that satisfy the following conditions:
//	- public method
//	- two arguments, both pointers to structs
//	- one return value of type os.Error
// It returns an error if the receiver is not suitable.
func Add(rcvr interface{}) os.Error {
	return server.add(rcvr)
}

// ServeConn runs the server on a single connection.  When the connection
// completes, service terminates.
func ServeConn(conn io.ReadWriteCloser) {
	go server.input(conn)
}

// Accept accepts connections on the listener and serves requests
// for each incoming connection.
func Accept(lis net.Listener) {
	server.accept(lis)
}

// Can connect to RPC service using HTTP CONNECT to rpcPath.
var rpcPath string = "/_goRPC_"
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
// It is still necessary to call http.Serve().
func HandleHTTP() {
	http.Handle(rpcPath, http.HandlerFunc(serveHTTP));
}
