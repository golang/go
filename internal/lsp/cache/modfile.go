package cache

import (
	"context"
	"go/token"
)

// modFile holds all of the information we know about a mod file.
type modFile struct {
	fileBase
}

func (*modFile) GetToken(context.Context) *token.File { return nil }
func (*modFile) setContent(content []byte)            {}
func (*modFile) filename() string                     { return "" }
func (*modFile) isActive() bool                       { return false }
