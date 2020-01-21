package cmd

import (
	"context"
	"io"
	"regexp"
	"testing"
	"time"

	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/telemetry/log"
)

type fakeServer struct {
	protocol.Server
	client protocol.Client
}

func (s *fakeServer) DidOpen(ctx context.Context, params *protocol.DidOpenTextDocumentParams) error {
	// Our instrumentation should cause this message to be logged back to the LSP
	// client.
	log.Print(ctx, "ping")
	return nil
}

type fakeClient struct {
	protocol.Client

	logs chan string
}

func (c *fakeClient) LogMessage(ctx context.Context, params *protocol.LogMessageParams) error {
	c.logs <- params.Message
	return nil
}

func TestClientLogging(t *testing.T) {
	server := &fakeServer{}
	client := &fakeClient{logs: make(chan string)}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Bind our fake client and server.
	// sReader and sWriter read from and write to the server. cReader and cWriter
	// read from and write to the client.
	sReader, sWriter := io.Pipe()
	cReader, cWriter := io.Pipe()
	close := func() {
		failOnErr := func(err error) {
			if err != nil {
				t.Fatal(err)
			}
		}
		failOnErr(sReader.Close())
		failOnErr(cReader.Close())
		failOnErr(sWriter.Close())
		failOnErr(cWriter.Close())
	}
	defer close()
	serverStream := jsonrpc2.NewStream(sReader, cWriter)
	// The returned client dispatches to the client, but it is already stored
	// in the context by NewServer, so we can ignore it.
	serverCtx, serverConn, _ := protocol.NewServer(ctx, serverStream, server)
	serverConn.AddHandler(&handler{})
	clientStream := jsonrpc2.NewStream(cReader, sWriter)
	clientCtx, clientConn, serverDispatch := protocol.NewClient(ctx, clientStream, client)

	go clientConn.Run(clientCtx)
	go serverConn.Run(serverCtx)
	serverDispatch.DidOpen(ctx, &protocol.DidOpenTextDocumentParams{})

	select {
	case got := <-client.logs:
		want := "ping"
		matched, err := regexp.MatchString(want, got)
		if err != nil {
			t.Fatal(err)
		}
		if !matched {
			t.Errorf("got log %q, want a log containing %q", got, want)
		}
	case <-time.After(1 * time.Second):
		t.Error("timeout waiting for client log")
	}
}
