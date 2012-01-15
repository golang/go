// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package ssh implements an SSH client and server.

SSH is a transport security protocol, an authentication protocol and a
family of application protocols. The most typical application level
protocol is a remote shell and this is specifically implemented.  However,
the multiplexed nature of SSH is exposed to users that wish to support
others.

An SSH server is represented by a ServerConfig, which holds certificate
details and handles authentication of ServerConns.

	config := new(ssh.ServerConfig)
	config.PubKeyCallback = pubKeyAuth
	config.PasswordCallback = passwordAuth

	pemBytes, err := ioutil.ReadFile("id_rsa")
	if err != nil {
		panic("Failed to load private key")
	}
	err = config.SetRSAPrivateKey(pemBytes)
	if err != nil {
		panic("Failed to parse private key")
	}

Once a ServerConfig has been configured, connections can be accepted.

	listener := Listen("tcp", "0.0.0.0:2022", config)
	sConn, err := listener.Accept()
	if err != nil {
		panic("failed to accept incoming connection")
	}
	if err := sConn.Handshake(conn); err != nil {
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
		channel.Reject(UnknownChannelType, "unknown channel type")
		return
	}
	channel.Accept()

	term := terminal.NewTerminal(channel, "> ")
	serverTerm := &ssh.ServerTerminal{
		Term: term,
		Channel: channel,
	}
	go func() {
		defer channel.Close()
		for {
			line, err := serverTerm.ReadLine()
			if err != nil {
				break
			}
			println(line)
		}
		return
	}()

To authenticate with the remote server you must pass at least one implementation of 
ClientAuth via the Auth field in ClientConfig.

	// password implements the ClientPassword interface
	type password string

	func (p password) Password(user string) (string, error) {
		return string(p), nil
	}

	config := &ssh.ClientConfig {
		User: "username",
		Auth: []ClientAuth {
			// ClientAuthPassword wraps a ClientPassword implementation
			// in a type that implements ClientAuth.
			ClientAuthPassword(password("yourpassword")),
		}
	}

An SSH client is represented with a ClientConn. Currently only the "password"
authentication method is supported.

	config := &ClientConfig{
		User: "username",
		Auth: []ClientAuth{ ... },
	}
	client, err := Dial("yourserver.com:22", config)

Each ClientConn can support multiple interactive sessions, represented by a Session.

	session, err := client.NewSession()

Once a Session is created, you can execute a single command on the remote side
using the Exec method.

	b := bytes.NewBuffer()
	session.Stdin = b
	if err := session.Run("/usr/bin/whoami"); err != nil {
		panic("Failed to exec: " + err.String())
	}
	fmt.Println(bytes.String())
	session.Close()
*/
package ssh
