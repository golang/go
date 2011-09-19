// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package ssh implements an SSH server.

SSH is a transport security protocol, an authentication protocol and a
family of application protocols. The most typical application level
protocol is a remote shell and this is specifically implemented.  However,
the multiplexed nature of SSH is exposed to users that wish to support
others.

An SSH server is represented by a Server, which manages a number of
ServerConnections and handles authentication.

	var s Server
	s.PubKeyCallback = pubKeyAuth
	s.PasswordCallback = passwordAuth

	pemBytes, err := ioutil.ReadFile("id_rsa")
	if err != nil {
		panic("Failed to load private key")
	}
	err = s.SetRSAPrivateKey(pemBytes)
	if err != nil {
		panic("Failed to parse private key")
	}

Once a Server has been set up, connections can be attached.

	var sConn ServerConnection
	sConn.Server = &s
	err = sConn.Handshake(conn)
	if err != nil {
		panic("failed to handshake")
	}

An SSH connection multiplexes several channels, which must be accepted themselves:


	for {
		channel, err := sConn.Accept()
		if err != nil {
			panic("error from Accept")
		}

		...
	}

Accept reads from the connection, demultiplexes packets to their corresponding
channels and returns when a new channel request is seen. Some goroutine must
always be calling Accept; otherwise no messages will be forwarded to the
channels.

Channels have a type, depending on the application level protocol intended. In
the case of a shell, the type is "session" and ServerShell may be used to
present a simple terminal interface.

	if channel.ChannelType() != "session" {
		c.Reject(UnknownChannelType, "unknown channel type")
		return
	}
	channel.Accept()

	shell := NewServerShell(channel, "> ")
	go func() {
		defer channel.Close()
		for {
			line, err := shell.ReadLine()
			if err != nil {
				break
			}
			println(line)
		}
		return
	}()
*/
package ssh
