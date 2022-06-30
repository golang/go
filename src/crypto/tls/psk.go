package tls

// Configuration for PSK cipher-suite. The client needs to provide a GetIdentity and GetKey functions to retrieve client id and pre-shared-key
type PSKConfig struct {
	// client-only - returns the client identity
	GetIdentity func() string

	// for server - returns the key associated to a client identity
	// for client - returns the key for this client
	GetKey func(identity string) ([]byte, error)
}
