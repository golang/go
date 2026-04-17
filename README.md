# Go — Privasys RA-TLS Fork

This is a fork of the [Go programming language](https://github.com/golang/go) maintained by [Privasys](https://privasys.org) to add **RA-TLS (Remote Attestation TLS)** support.

## What this fork adds

A custom TLS extension (`0xFFBB`) that carries a challenge nonce in the **ClientHello** and **CertificateRequest** messages. This enables RA-TLS flows where:

1. A TLS **client** embeds a challenge nonce in the ClientHello extension.
2. The TLS **server** (e.g. an SGX/TDX enclave) reads the nonce and binds it into the attestation quote's `report_data` field via `SHA-512(public_key_sha256 || nonce)`.
3. The verifier can confirm the quote was freshly generated for that specific TLS session.

### Changed files (relative to upstream `go1.25.7`)

| File | Change |
|------|--------|
| `src/crypto/tls/common.go` | Added `Config.RATLSChallenge` field and `ClientHelloInfo.RATLSChallenge` |
| `src/crypto/tls/handshake_client.go` | Copy `Config.RATLSChallenge` → `clientHelloMsg` |
| `src/crypto/tls/handshake_messages.go` | Marshal/unmarshal `extensionRATLS` in `clientHelloMsg` and `certificateRequestMsgTLS13` |
| `src/crypto/tls/handshake_client_tls13.go` | Propagate challenge from CertificateRequest to `CertificateRequestInfo` |
| `src/crypto/tls/handshake_server.go` | Propagate challenge from ClientHello to `ClientHelloInfo` |
| `src/crypto/tls/handshake_server_tls13.go` | Copy `Config.RATLSChallenge` to CertificateRequest |
| `src/crypto/tls/handshake_messages_test.go` | Unit tests for marshal/unmarshal round-trip |
| `src/crypto/tls/tls_test.go` | Config clone test for `RATLSChallenge` |

### Usage

```go
import "crypto/tls"

// Client side: send a challenge nonce
config := &tls.Config{
    RATLSChallenge:     nonce,       // 8–64 bytes
    InsecureSkipVerify: true,        // attestation replaces PKI
}
conn, err := tls.Dial("tcp", "enclave:443", config)

// Server side: read the challenge from ClientHelloInfo
config := &tls.Config{
    GetCertificate: func(hello *tls.ClientHelloInfo) (*tls.Certificate, error) {
        nonce := hello.RATLSChallenge // the challenge from the client
        // ... generate attestation quote binding this nonce ...
    },
}
```

## Upstream

This fork is based on the `release-branch.go1.25` branch (tag `go1.25.7`) of the upstream Go repository at https://github.com/golang/go.

Unless otherwise noted, the Go source files are distributed under the
BSD-style license found in the LICENSE file.

### Download and Install

#### Binary Distributions

Official binary distributions are available at https://go.dev/dl/.

After downloading a binary release, visit https://go.dev/doc/install
for installation instructions.

#### Install From Source

If a binary distribution is not available for your combination of
operating system and architecture, visit
https://go.dev/doc/install/source
for source installation instructions.

### Contributing

Go is the work of thousands of contributors. We appreciate your help!

To contribute, please read the contribution guidelines at https://go.dev/doc/contribute.

Note that the Go project uses the issue tracker for bug reports and
proposals only. See https://go.dev/wiki/Questions for a list of
places to ask questions about the Go language.

[rf]: https://reneefrench.blogspot.com/
[cc4-by]: https://creativecommons.org/licenses/by/4.0/
