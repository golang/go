package harness

import (
	"bytes"

	"github.com/caddyserver/caddy/v2"
	"github.com/caddyserver/caddy/v2/caddyconfig/caddyfile"
	"github.com/caddyserver/caddy/v2/caddyconfig/httpcaddyfile"
)

func harness(data []byte) int {
	if len(data) < 2 {
		return 0
	}

	s := make([]byte, len(data))
	copy(s, data)

	switch s[0] {
	case 0x00:
		_, err := caddy.ParseNetworkAddress(string(s[1:]))
		if err != nil {
			return 0
		}
		return 1
	case 0x01:
		caddyfile.Tokenize(s[1:], "Caddyfile")
	case 0x02:
		_, err := caddy.ParseDuration(string(s[1:]))
		if err != nil {
			return 0
		}
		return 1
	case 0x03:
		data := s[1:]
		caddy.NewReplacer().ReplaceAll(string(data), "")
		caddy.NewReplacer().ReplaceAll(caddy.NewReplacer().ReplaceAll(string(data), ""), "")
		caddy.NewReplacer().ReplaceAll(caddy.NewReplacer().ReplaceAll(string(data), ""), caddy.NewReplacer().ReplaceAll(string(data), ""))
		caddy.NewReplacer().ReplaceAll(string(data[:len(data)/2]), string(data[len(data[1:])/2:]))
		return 0
	case 0x04:
		formatted := caddyfile.Format(s)
		if bytes.Equal(formatted, caddyfile.Format(formatted)) {
			return 1
		}
		return 0
	case 0x05:
		addr, err := httpcaddyfile.ParseAddress(string(s[1:]))
		if err != nil {
			if addr == (httpcaddyfile.Address{}) {
				return 1
			}
			return 0
		}
		return 1

		/*
			todo: not exported directly but tested by oss-fuzz
				case 0x06:
					_, _, err := templates.extractFrontMatter(string(s[1:]))
					if err != nil {
						return 0
					}
					return 1
		*/
	}
	return 0
}
